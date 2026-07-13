from search import search_prompt

def main():

    print("Digite uma pergunta:")
    print("Para sair, digite 'sair', 'exit' ou 'quit'")
    
    while True:
        question = input("> ")

        if question.lower() in ["sair", "exit", "quit"]:
            break

        resposta = search_prompt(question)

        if not resposta:
            print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
            return

        print(resposta)

if __name__ == "__main__":
    main()