from llm import LLMInferenceEngine, ChatModel
from transformers import AutoTokenizer
import argparse
from colorama import Fore, init

init(autoreset=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--backend", type=str, default="vllm")
    parser.add_argument("--debug_mode", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    engine = LLMInferenceEngine(args.model_id, args.backend)
    chatbot = ChatModel(engine)
    if args.debug_mode:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    labels = ["Model ID", "backend", "You", "Bot"]

    max_label_length = max(len(label) for label in labels)
    
    delimiter = "=" * 40
    print(Fore.YELLOW + "\n" + delimiter)
    print(Fore.YELLOW + f"{'Model ID'.ljust(max_label_length)}: {args.model_id}")
    print(Fore.YELLOW + f"{'backend'.ljust(max_label_length)}: {args.backend}")
    print(Fore.YELLOW + "Type 'exit' to quit the chat.")
    print(Fore.YELLOW + delimiter + "\n")

    while True:
        user_input = input(f"{'You'.ljust(max_label_length)}: ")
        if user_input == "exit":
            break
        if user_input == "clear":
            chatbot.clear_dialogue()
            print(Fore.LIGHTBLACK_EX + f"Dialogue history cleared: {chatbot.get_full_dialogue()}")
            continue
        chat_model_response = chatbot.send_and_get(user_input)
        print(Fore.LIGHTCYAN_EX + f"{'Bot'.ljust(max_label_length)}: {chat_model_response.response}")

        if args.debug_mode:
            print(Fore.LIGHTBLACK_EX + f"[DEBUG MODE] INPUT PROMPT: {repr(chat_model_response.prompt)}")
            print(Fore.LIGHTBLACK_EX + f"[DEBUG MODE] # of INPUT TOKENS: {len(tokenizer.encode(chat_model_response.prompt))}")
            print(Fore.LIGHTBLACK_EX + f"[DEBUG MODE] FULL DIALOGUE LIST: {chatbot.get_full_dialogue()}")

        print()

if __name__ == "__main__":
    main()