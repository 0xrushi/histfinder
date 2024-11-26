from pathlib import Path
import os
from time import sleep
import pandas as pd
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
import argparse
import subprocess
import pyautogui

load_dotenv()

# def extract_paths_from_command(command):
#     # Split command into words and filter potential paths
#     words = command.split()
#     paths = []
    
#     for word in words:
#         if word.startswith(('http://', 'https://')):
#             continue
            
#         if ('/' in word or '~' in word) and not word.startswith('-'):
#             expanded_path = os.path.expanduser(word)
#             resolved_path = os.path.abspath(expanded_path)
#             paths.append(resolved_path)
    
#     return paths

def read_zsh_history(file_path=None):
    history_path = Path(file_path if file_path else os.path.expanduser("~/.zsh_history"))
    if not history_path.exists():
        return []
    commands = []
    try:
        with open(history_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.strip().split(';', 1)
                if len(parts) > 1:
                    command = parts[1].strip()
                    commands.append(command)
    
    except Exception as e:
        print(f"Error reading history file: {e}")
        return []
        
    return list(set(commands))

async def rebuild_description(command: str) -> str:
    """Generate a description of what a command does using Groq API."""

    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are a helpful assistant that explains command line commands.
    Please explain what this command does in a clear and concise way in one sentence maximum:
    
    Command: {command}
    
    Explanation:"""

    # Generate response
    response = await client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt
        }],
        model="llama3-8b-8192",
        temperature=0.3,
        max_tokens=256,
    )

    return response.choices[0].message.content.strip()

def load_existing_results(output_file: str) -> tuple[pd.DataFrame, set[str]]:
    """Load existing results and return processed commands."""
    try:
        df = pd.read_csv(output_file)
        return df, set(df['command'].tolist())
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=['command', 'description']), set()

async def generate_db(commands, clean_commands, processed_commands):
    """Generate a database of commands with their descriptions.
    
    Args:
        commands (list): List of shell commands to process
        clean_commands (list): List to store processed commands with descriptions
        processed_commands (set): Set of commands that have already been processed
    
    The function processes each command by:
    1. Generating a description using the Groq API
    2. Storing the command and description in clean_commands
    3. Saving progress to CSV after each successful command
    """
    for command in commands:
        # if "docker" in command and command not in processed_commands:
        if True:
            try:
                print(command)
                description = await rebuild_description(command)
                clean_commands.append({
                    "command": command,
                    "description": description
                })
                # Save progress after each successful command
                df = pd.DataFrame(clean_commands)
                df.to_csv(output_file, index=None)
            except Exception as e:
                print(f"Error processing command '{command}': {e}")
                continue

    df = pd.DataFrame(clean_commands)
    df.to_csv(output_file, index=None)

async def search_commands(query: str, vectorstore):
    """Search for commands matching the query."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    results = retriever.invoke(query)
    
    for doc in results:
        res = json.loads(doc.page_content)
        return res['command']
    return None

async def main():
    parser = argparse.ArgumentParser(
        prog='histfinder',
        description='Search through command history using natural language'
    )
    parser.add_argument('query', nargs='?', help='Natural language query to search for commands')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the command database')
    parser.add_argument('-f', '--file', help='Specify an alternative history file to read from')
    args = parser.parse_args()

    output_file = "clean_commands_with_description_v0.csv"
    
    if args.rebuild:
        commands = read_zsh_history(args.file)
        existing_df, processed_commands = load_existing_results(output_file)
        clean_commands = existing_df.to_dict('records')
        await generate_db(commands, clean_commands, processed_commands)
        print("Database rebuilt successfully")
        return

    if not args.query:
        parser.print_help()
        return

    try:
        existing_df, _ = load_existing_results(output_file)
        clean_commands = existing_df.to_dict('records')
        
        documents = [
            Document(
                page_content=json.dumps({
                    "command": row['command'],
                    "description": row['description']
                }),
                metadata={"command": row['command']}
            )
            for row in clean_commands
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=5)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        
        result = await search_commands(args.query, vectorstore)
        if result:
            print(f"Found command: \033[94m{result}\033[0m\nDo you want to run the command?.")
            response = input("(y/n): ").lower()
            if response in ['y', 'yes']:
                process = subprocess.Popen('pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
                process.communicate(result.encode('utf-8'))
                print("")
                sleep(0.2)
                pyautogui.keyDown('command')
                pyautogui.keyDown('v')
        else:
            print("No matching command ")
            
    except Exception as e:
        print(f"Error: {e}")

def run_async_main():
    asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())
    
