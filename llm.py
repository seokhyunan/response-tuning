import torch
import concurrent.futures
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.utils import ModelOutput
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletion
from colorama import Fore, init
from typing import List, Union, Callable
import time
from tqdm import tqdm
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
import gc

init(autoreset=True)


class UniversalGenParams:
    def __init__(
        self,
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        seed: int = 0,
        stop: List[str] = None,
    ):
        self.n = n
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.stop = stop

    def __dict__(self):
        return {
            "n": self.n,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }

    def get_hf_params(self):
        return {
            "num_return_sequences": self.n,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "do_sample": True if self.temperature > 0 and self.top_p < 1 else False, # hf specific
            "output_logits": True, # hf specific
            "return_dict_in_generate": True, # hf specific
            "stop_strings": self.stop, # hf specific
        }

    def get_vllm_params(self):
        return SamplingParams(
            n=self.n,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            logprobs=True, # vllm specific
            prompt_logprobs=True, # vllm specific
            stop=self.stop,
        )
    
    def get_openai_params(self):
        return {
            "n": self.n,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "logprobs": True, # openai specific
            "stop": self.stop,
        }


class GenerationArgs:
    def __init__(
        self,
        engine_input: Union[str, List[str], List[List[str]]] = None,
        gen_params: UniversalGenParams = UniversalGenParams(),
        response_format: Union[ModelMetaclass, BaseModel, dict] = None,
        is_multi_turn_input: bool = False,
        is_batch_input: bool = False,
        apply_chat_template: bool = True,
        use_system_prompt: bool = False,
        add_generation_prompt: bool = True,
        user_role: str = "user",
        model_role: str = "assistant",
        sys_role: str = "system",
        chat_prompt_processor: Callable = None,
        chat_prompt_processor_kwargs: dict = {},
    ):
        self.engine_input = engine_input
        self.gen_params = gen_params
        self.response_format = response_format
        self.is_multi_turn_input = is_multi_turn_input
        self.is_batch_input = is_batch_input
        self.apply_chat_template = apply_chat_template
        self.use_system_prompt = use_system_prompt
        self.add_generation_prompt = add_generation_prompt
        self.user_role = user_role
        self.model_role = model_role
        self.sys_role = sys_role
        self.chat_prompt_processor = chat_prompt_processor
        self.chat_prompt_processor_kwargs = chat_prompt_processor_kwargs

    def __dict__(self):
        return {
            "engine_input": self.engine_input,
            "gen_params": dict(self.gen_params),
            "is_multi_turn_input": self.is_multi_turn_input,
            "is_batch_input": self.is_batch_input,
            "apply_chat_template": self.apply_chat_template,
            "use_system_prompt": self.use_system_prompt,
            "add_generation_prompt": self.add_generation_prompt,
            "user_role": self.user_role,
            "model_role": self.model_role,
            "sys_role": self.sys_role,
        }


class LLMInferenceOutput:
    def __init__(
        self,
        output_seqs=None,
        output_objects=None,
        input_prompt=None,
        logits=None,
        finish_reasons=None,
        logprobs=None,
        prompt_logprobs=None,
        cumulative_output_logprobs=None,
        latency=None,
        usage=None,
    ):
        self.input_prompt = input_prompt

        assert isinstance(output_seqs, list), "output_seqs must be a list of sequences"
        self.output_seqs = output_seqs
        self.output_objects = output_objects # for structured output (openai)

        # for hf
        self.logits = logits

        # for vllm
        self.finish_reasons = finish_reasons
        self.logprobs = logprobs 
        self.prompt_logprobs = prompt_logprobs
        self.cumulative_output_logprobs = cumulative_output_logprobs
        self.latency = latency

        # for openai
        self.usage = usage


class LLMInferenceEngine:
    def __init__(self, model_id, backend, custom_chat_template=None, backend_kwargs={}):
        self.model_id = model_id
        self.backend = backend
        self._load_model(custom_chat_template, backend_kwargs)
    
    def _load_model(self, custom_chat_template, backend_kwargs):
        if self.backend == "hf":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                **backend_kwargs,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        elif self.backend == "vllm":
            self.model = LLM(self.model_id, trust_remote_code=True, **backend_kwargs)
            self.tokenizer = self.model.get_tokenizer()
        elif self.backend == "openai":
            self.model = OpenAI()
            self.tokenizer = None
            self.num_openai_workers = 10
        elif self.backend == "vllm-openai":
            base_url = backend_kwargs.get("base_url")
            if base_url is None:
                base_url = "http://localhost:8000/v1"
            print(Fore.YELLOW + f"[INFO] Using OpenAI API at {base_url}")
            print(Fore.YELLOW + f"[INFO] API key: {self.model_id}")
            self.model = OpenAI(base_url=base_url, api_key=self.model_id, timeout=None)
            self.tokenizer = None
            self.num_openai_workers = 500
        else:
            raise ValueError("Invalid engine")
        
        if custom_chat_template is not None and self.backend in ("hf", "vllm"):
            self.tokenizer.chat_template = custom_chat_template
    
    def generate(self, gen_args: GenerationArgs) -> List[LLMInferenceOutput]:
        if not gen_args.is_multi_turn_input and gen_args.use_system_prompt:
            print(Fore.YELLOW + "[WARNING] use_system_prompt is True but is_multi_turn_input is False. Forcing is_multi_turn_input to True.")
            gen_args.is_multi_turn_input = True
        
        if self.backend in ("hf", "vllm"):
            if gen_args.is_multi_turn_input and not gen_args.apply_chat_template:
                print(Fore.YELLOW + "[WARNING] is_multi_turn_input is True but apply_chat_template is False. Forcing apply_chat_template to True.")
                gen_args.apply_chat_template = True
            if self.tokenizer.chat_template is not None and not gen_args.apply_chat_template:
                print(Fore.YELLOW + "[WARNING] Chat template is defined but apply_chat_template is False. Forcing apply_chat_template to True.")
                gen_args.apply_chat_template = True

        if self.backend == "hf":
            return self._hf_generate(gen_args)
        elif self.backend == "vllm":
            return self._vllm_generate(gen_args)
        elif self.backend in ("openai", "vllm-openai"):
            return self._openai_generate(gen_args)
        else:
            raise ValueError("Invalid engine")

    def _hf_generate(self, gen_args: GenerationArgs) -> List[LLMInferenceOutput]:
        set_seed(0)
        prompts = self._hf_vllm_input_preprocess(gen_args)
        gen_params = gen_args.gen_params.get_hf_params()
        model_inputs = [self.tokenizer(prompt, return_tensors="pt").to(self.model.device) for prompt in prompts]
        model_outputs = self._hf_inference(model_inputs, gen_params=gen_params)
        outputs = self._hf_parse_output(model_outputs, prompts)
        return outputs
    
    def _vllm_generate(self, gen_args: GenerationArgs) -> List[LLMInferenceOutput]:
        prompts = self._hf_vllm_input_preprocess(gen_args)
        gen_params = gen_args.gen_params.get_vllm_params()
        model_outputs = self._vllm_inference(prompts, gen_params=gen_params)
        assert len(model_outputs) == len(prompts), "Number of inputs and outputs are not the same."
        outputs = self._vllm_parse_output(model_outputs)
        return outputs
    
    def _openai_generate(self, gen_args: GenerationArgs) -> List[LLMInferenceOutput]:
        prompts = self._openai_input_preprocess(gen_args)
        gen_params = gen_args.gen_params.get_openai_params()
        if gen_args.is_batch_input:
            model_outputs = self._openai_parallel_inference(prompts, gen_params=gen_params, response_format=gen_args.response_format)
        else:
            model_outputs = self._openai_inference(prompts, gen_params=gen_params, response_format=gen_args.response_format)
        outputs = self._openai_parse_output(model_outputs, prompts)
        return outputs
    
    def _hf_vllm_input_preprocess(self, gen_args: GenerationArgs) -> List[str]:
        if (gen_args.is_multi_turn_input and gen_args.is_batch_input) or (not gen_args.is_multi_turn_input and gen_args.is_batch_input and gen_args.apply_chat_template):
            chats = [self._construct_chat(chat, use_system_prompt=gen_args.use_system_prompt, user_role=gen_args.user_role, model_role=gen_args.model_role, sys_role=gen_args.sys_role) for chat in gen_args.engine_input]
            prompts = [
                self._apply_prompt_level_chat_template(
                    chat,
                    add_generation_prompt=gen_args.add_generation_prompt,
                    chat_prompt_processor=gen_args.chat_prompt_processor,
                    chat_prompt_processor_kwargs=gen_args.chat_prompt_processor_kwargs
                ) for chat in chats
            ]
        elif (gen_args.is_multi_turn_input and not gen_args.is_batch_input) or (not gen_args.is_multi_turn_input and not gen_args.is_batch_input and gen_args.apply_chat_template):
            chat = self._construct_chat(gen_args.engine_input, use_system_prompt=gen_args.use_system_prompt, user_role=gen_args.user_role, model_role=gen_args.model_role, sys_role=gen_args.sys_role)
            prompts = [
                self._apply_prompt_level_chat_template(
                    chat,
                    add_generation_prompt=gen_args.add_generation_prompt,
                    chat_prompt_processor=gen_args.chat_prompt_processor,
                    chat_prompt_processor_kwargs=gen_args.chat_prompt_processor_kwargs
                )
            ]
        else:
            prompts = gen_args.engine_input if isinstance(gen_args.engine_input, list) else [gen_args.engine_input]
        return prompts
    
    def _openai_input_preprocess(self, gen_args: GenerationArgs) -> List[str]:
        if (gen_args.is_multi_turn_input and gen_args.is_batch_input) or (not gen_args.is_multi_turn_input and gen_args.is_batch_input):
            prompts = [self._construct_chat(chat, use_system_prompt=gen_args.use_system_prompt, user_role=gen_args.user_role, model_role=gen_args.model_role, sys_role=gen_args.sys_role) for chat in gen_args.engine_input]
        elif (gen_args.is_multi_turn_input and not gen_args.is_batch_input) or (not gen_args.is_multi_turn_input and not gen_args.is_batch_input):
            prompts = self._construct_chat(gen_args.engine_input, use_system_prompt=gen_args.use_system_prompt, user_role=gen_args.user_role, model_role=gen_args.model_role, sys_role=gen_args.sys_role)
        else:
            prompts = gen_args.engine_input if isinstance(gen_args.engine_input, list) else [gen_args.engine_input]
        return prompts

    def _hf_inference(self, hf_model_inputs, gen_params) -> List[ModelOutput]:
        model_outputs = []
        for model_input in tqdm(hf_model_inputs, desc="HF Inference"):
            model_outputs.append(self.model.generate(**model_input, **gen_params))
        return model_outputs

    def _vllm_inference(self, prompts, gen_params) -> List[RequestOutput]: 
        model_output = self.model.generate(prompts, gen_params)
        return model_output
    
    def _openai_inference(self, prompt, gen_params, response_format) -> ChatCompletion:
        attempts = 0
        while True:
            try:
                model_output = self.model.chat.completions.create(
                    model=self.model_id,
                    messages=prompt,
                    response_format=response_format,
                    **gen_params
                ) if not isinstance(response_format, (BaseModel, ModelMetaclass)) else self.model.beta.chat.completions.parse(
                    model=self.model_id,
                    messages=prompt,
                    response_format=response_format,
                    **gen_params
                )
                break
            except OpenAIError as e:
                print(Fore.RED + f"[ERROR] OpenAIError: {e}")
                if "Please try again with a different prompt." in str(e):
                    model_output = None
                    break
                else:
                    if attempts < 50:
                        print(Fore.YELLOW + f"[INFO] Retrying in {(5 * attempts) % 60} seconds (attempt {attempts + 1})")
                        attempts += 1
                        time.sleep((5 * attempts) % 60)
                    else:
                        print(Fore.RED + f"[ERROR] Max attempts reached")
                        model_output = None
                        break
        return model_output
        
    def _openai_parallel_inference(self, prompts, gen_params, response_format) -> List[ChatCompletion]:
        # Initialize a list to store results in the correct order
        results = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(prompts), self.num_openai_workers)) as executor:
            futures = {executor.submit(self._openai_inference, prompt, gen_params, response_format): idx for idx, prompt in enumerate(prompts)}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="OpenAI Parallel Inference"):
                result = future.result()
                idx = futures[future]
                results[idx] = result
            return results
        
    def _hf_parse_output(self, model_outputs, input_prompts) -> List[LLMInferenceOutput]:
        def process_single_output(model_output, input_prompt):
            input_token_length = self.tokenizer(input_prompt, return_tensors="pt")["input_ids"].shape[-1]
            output_seqs = [seq[input_token_length:] for seq in model_output.sequences]
            output_seqs = [self._decode_tokens(seq) for seq in output_seqs]
            logits = model_output.logits
            return LLMInferenceOutput(output_seqs=output_seqs, output_objects=model_output, input_prompt=input_prompt, logits=logits)
        if isinstance(model_outputs, list):
            outputs = [process_single_output(model_output, input_prompt) for model_output, input_prompt in zip(model_outputs, input_prompts)]
            return outputs
        elif isinstance(model_outputs, ModelOutput):
            return [process_single_output(model_outputs, input_prompts)]
    
    def _vllm_parse_output(self, vllm_outputs) -> List[LLMInferenceOutput]:
        batch_size = len(vllm_outputs)
        outputs =[]
        for idx in range(batch_size):
            request_output = vllm_outputs[idx]
            latency=request_output.metrics.finished_time - request_output.metrics.first_scheduled_time,
            prompt_logprobs=request_output.prompt_logprobs

            generation_output = sorted(request_output.outputs, key=lambda x: x.index)
            seqs, logprobs, cumulative_output_logprobs, finish_reasons = zip(*[(s.text, s.logprobs, s.cumulative_logprob, s.finish_reason) for s in generation_output])
            seqs, logprobs, cumulative_output_logprobs, finish_reasons = list(seqs), list(logprobs), list(cumulative_output_logprobs), list(finish_reasons)

            outputs.append(
                LLMInferenceOutput(
                    output_seqs=seqs,
                    output_objects=generation_output,
                    input_prompt=request_output.prompt,
                    logprobs=logprobs,
                    cumulative_output_logprobs=cumulative_output_logprobs,
                    finish_reasons=finish_reasons,
                    latency=latency,
                    prompt_logprobs=prompt_logprobs
                )
            )
        return outputs

    def _openai_parse_output(self, client_outputs, prompts) -> List[LLMInferenceOutput]:
        def process_single_output(client_output, prompt):
            usage = client_output.usage
            generation_outputs = sorted(client_output.choices, key=lambda x: x.index)
            output_seqs, logprobs, finish_reasons = [], [], []
            for i in range(len(generation_outputs)):
                if hasattr(generation_outputs[i].message, 'parsed'):
                    if generation_outputs[i].message.parsed is not None:
                        output_seqs.append(generation_outputs[i].message.parsed)
                    else:
                        output_seqs.append(generation_outputs[i].message.content)
                else:
                    output_seqs.append(generation_outputs[i].message.content)
                logprobs.append(generation_outputs[i].logprobs)
                finish_reasons.append(generation_outputs[i].finish_reason)
            return LLMInferenceOutput(
                output_seqs=output_seqs,
                output_objects=client_output,
                usage=usage,
                input_prompt=prompt,
                logprobs=logprobs,
                finish_reasons=finish_reasons
            )
        if isinstance(client_outputs, list):
            outputs = [process_single_output(client_output, prompt) for client_output, prompt in zip(client_outputs, prompts)]
            return outputs
        elif isinstance(client_outputs, ChatCompletion):
            return [process_single_output(client_outputs, prompts)]
        else:
            raise ValueError("Invalid output type")

    def _construct_chat(self,
            msg_list, 
            use_system_prompt=False, 
            user_role="user", 
            model_role="assistant", 
            sys_role="system"
        ) -> List[dict]:
        if isinstance(msg_list, str):
            msg_list = [msg_list]
        def format_turn(role, message):
            return {"role": role, "content": message}
        roles = [user_role, model_role]
        chat = []
        if use_system_prompt:
            chat.append(format_turn(sys_role, msg_list.pop(0)))
        for idx, message in enumerate(msg_list):
            chat.append(format_turn(roles[idx % 2], message))
        return chat

    def _apply_prompt_level_chat_template(self, chat, add_generation_prompt=True, chat_prompt_processor=None, chat_prompt_processor_kwargs=None) -> str:
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=add_generation_prompt)
        if chat_prompt_processor is not None:
            prompt = chat_prompt_processor(prompt, **chat_prompt_processor_kwargs)
        return prompt
    
    def _decode_tokens(self, tokens, skip_special_tokens=True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def _shutdown(self):
        if self.backend == "vllm":
            destroy_model_parallel()
            destroy_distributed_environment()
            del self.model.llm_engine.model_executor
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
        elif self.backend == "hf":
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

    def __del__(self):
        self._shutdown()


class ChatModel:
    def __init__(self, engine):
        self.engine = engine
        self.chat_generation_args = GenerationArgs(
            engine_input=[],
            gen_params=UniversalGenParams(max_new_tokens=2048, temperature=0.0),
            is_multi_turn_input=True,
            is_batch_input=False,
            apply_chat_template=True,
            use_system_prompt=False,
        )
    
    def send_and_get(self, msg):
        self.chat_generation_args.engine_input.append(msg)
        engine_output = self.engine.generate(self.chat_generation_args)
        model_response, input_prompt = engine_output[0].output_seqs[0].strip(), engine_output[0].input_prompt
        self.chat_generation_args.engine_input.append(model_response)
        return ChatModelResponse(model_response, input_prompt)

    def get_full_dialogue(self):
        return self.chat_generation_args.engine_input
    
    def clear_dialogue(self):
        self.chat_generation_args.engine_input = []

class ChatModelResponse:
    def __init__(self, response, prompt=None):
        self.response = response
        self.prompt = prompt