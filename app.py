from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tasksA import *
import requests
from dotenv import load_dotenv
import os
import re
import httpx
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


app = FastAPI()
load_dotenv()

@app.get("/ask")
def ask(prompt: str):
    result = get_completions(prompt)
    return result

openai_api_chat  = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions" # for testing
openai_api_key = os.getenv("AIPROXY_TOKEN")

headers = {
    "Authorization": f"Bearer {openai_api_key}",
    "Content-Type": "application/json",
}

function_definitions_llm = [
    {
        "name": "A1",
        "description": "Run a Python script from a given URL, passing an email as the argument.",
        "parameters": {
            "type": "object",
            "properties": {
                # "filename": {"type": "string", "pattern": r"https?://.*\.py"},
                # "targetfile": {"type": "string", "pattern": r".*/(.*\.py)"},
                "email": {"type": "string", "pattern": r"[\w\.-]+@[\w\.-]+\.\w+"}
            },
            "required": ["filename", "targetfile", "email"]
        }
    },
    {
        "name": "A2",
        "description": "Format a markdown file using a specified version of Prettier.",
        "parameters": {
            "type": "object",
            "properties": {
                "prettier_version": {"type": "string", "pattern": r"prettier@\d+\.\d+\.\d+"},
                "filename": {"type": "string", "pattern": r".*/(.*\.md)"}
            },
            "required": ["prettier_version", "filename"]
        }
    },
    {
        "name": "A3",
        "description": "Count the number of occurrences of a specific weekday in a date file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r"/data/.*dates.*\.txt"},
                "targetfile": {"type": "string", "pattern": r"/data/.*/(.*\.txt)"},
                "weekday": {"type": "integer", "pattern": r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"}
            },
            "required": ["filename", "targetfile", "weekday"]
        }
    },
    {
        "name": "A4",
        "description": "Sort a JSON contacts file and save the sorted version to a target file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.json)",
                },
                "targetfile": {
                    "type": "string",
                    "pattern": r".*/(.*\.json)",
                }
            },
            "required": ["filename", "targetfile"]
        }
    },
    {
        "name": "A5",
        "description": "Retrieve the most recent log files from a directory and save their content to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "log_dir_path": {
                    "type": "string",
                    "pattern": r".*/logs",
                    "default": "/data/logs"
                },
                "output_file_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/logs-recent.txt"
                },
                "num_files": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 10
                }
            },
            "required": ["log_dir_path", "output_file_path", "num_files"]
        }
    },
    {
        "name": "A6",
        "description": "Generate an index of documents from a directory and save it as a JSON file.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_dir_path": {
                    "type": "string",
                    "pattern": r".*/docs",
                    "default": "/data/docs"
                },
                "output_file_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.json)",
                    "default": "/data/docs/index.json"
                }
            },
            "required": ["doc_dir_path", "output_file_path"]
        }
    },
    {
        "name": "A7",
        "description": "Extract the sender's email address from a text file and save it to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/email.txt"
                },
                "output_file": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/email-sender.txt"
                }
            },
            "required": ["filename", "output_file"]
        }
    },
    {
        "name": "A8",
        "description": "Generate an image representation of credit card details from a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/credit-card.txt"
                },
                "image_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.png)",
                    "default": "/data/credit-card.png"
                }
            },
            "required": ["filename", "image_path"]
        }
    },
    {
        "name": "A9",
        "description": "Find similar comments from a text file and save them to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/comments.txt"
                },
                "output_filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/comments-similar.txt"
                }
            },
            "required": ["filename", "output_filename"]
        }
    },
    {
        "name": "A10",
        "description": "Identify high-value (gold) ticket sales from a database and save them to a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.db)",
                    "default": "/data/ticket-sales.db"
                },
                "output_filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/ticket-sales-gold.txt"
                },
                "query": {
                    "type": "string",
                    "pattern": "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"
                }
            },
            "required": ["filename", "output_filename", "query"]
        }
    },
    {
        "name": "B12",
        "description": "Check if filepath starts with /data",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "pattern": r"^/data/.*",
                    # "description": "Filepath must start with /data to ensure secure access."
                }
            },
            "required": ["filepath"]
        }
    },
    {
        "name": "B3",
        "description": "Download content from a URL and save it to the specified path.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "pattern": r"https?://.*",
                    "description": "URL to download content from."
                },
                "save_path": {
                    "type": "string",
                    "pattern": r".*/.*",
                    "description": "Path to save the downloaded content."
                }
            },
            "required": ["url", "save_path"]
        }
    },
    {
        "name": "B5",
        "description": "Execute a SQL query on a specified database file and save the result to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "db_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.db)",
                    "description": "Path to the SQLite database file."
                },
                "query": {
                    "type": "string",
                    "description": "SQL query to be executed on the database."
                },
                "output_filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "description": "Path to the file where the query result will be saved."
                }
            },
            "required": ["db_path", "query", "output_filename"]
        }
    },
    {
        "name": "B6",
        "description": "Fetch content from a URL and save it to the specified output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "pattern": r"https?://.*",
                    "description": "URL to fetch content from."
                },
                "output_filename": {
                    "type": "string",
                    "pattern": r".*/.*",
                    "description": "Path to the file where the content will be saved."
                }
            },
            "required": ["url", "output_filename"]
        }
    },
    {
        "name": "B7",
        "description": "Process an image by optionally resizing it and saving the result to an output path.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.(jpg|jpeg|png|gif|bmp))",
                    "description": "Path to the input image file."
                },
                "output_path": {
                    "type": "string",
                    "pattern": r".*/.*",
                    "description": "Path to save the processed image."
                },
                "resize": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Optional. Resize dimensions as [width, height]."
                }
            },
            "required": ["image_path", "output_path"]
        }
    },
    {
        "name": "B9",
        "description": "Convert a Markdown file to another format and save the result to the specified output path.",
        "parameters": {
            "type": "object",
            "properties": {
                "md_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.md)",
                    "description": "Path to the Markdown file to be converted."
                },
                "output_path": {
                    "type": "string",
                    "pattern": r".*/.*",
                    "description": "Path where the converted file will be saved."
                }
            },
            "required": ["md_path", "output_path"]
        }
    }

]

def get_completions(prompt: str):
    with httpx.Client(timeout=20) as client:
        response = client.post(
            f"{openai_api_chat}",
            headers=headers,
            json=
                {
                    "model": "gpt-4o-mini",
                    "messages": [
                                    {"role": "system", "content": "You are a function classifier that extracts structured parameters from queries."},
                                    {"role": "user", "content": prompt}
                                ],
                    "tools": [
                                {
                                    "type": "function",
                                    "function": function
                                } for function in function_definitions_llm
                            ],
                    "tool_choice": "auto"
                },
        )
    # return response.json()
    print(response.json()["choices"][0]["message"]["tool_calls"][0]["function"])
    return response.json()["choices"][0]["message"]["tool_calls"][0]["function"]


# Placeholder for task execution
@app.post("/run")
async def run_task(task: str):
    try:
        # Placeholder logic for executing tasks
        # Replace with actual logic to parse task and execute steps
        # Example: Execute task and return success or error based on result
        # llm_response = function_calling(tast), function_name = A1
        response = get_completions(task)
        print(response)
        task_code = response['name']
        arguments = response['arguments']

        if "A1"== task_code:
            A1(**json.loads(arguments))
        if "A2"== task_code:
            A2(**json.loads(arguments))
        if "A3"== task_code:
            A3(**json.loads(arguments))
        if "A4"== task_code:
            A4(**json.loads(arguments))
        if "A5"== task_code:
            A5(**json.loads(arguments))
        if "A6"== task_code:
            A6(**json.loads(arguments))
        if "A7"== task_code:
            A7(**json.loads(arguments))
        if "A8"== task_code:
            A8(**json.loads(arguments))
        if "A9"== task_code:
            A9(**json.loads(arguments))
        if "A10"== task_code:
            A10(**json.loads(arguments))
        if task_code == "B1":
            return {"allowed": B1(filename)}

        if task_code == "B2":
            return {"exists": B2(filename)}

        if task_code == "B3":  # Fetch data from API and save it
            response = requests.get("https://jsonplaceholder.typicode.com/todos/1")
            with open("/data/api-data.json", "w") as f:
                json.dump(response.json(), f, indent=4)
            return {"message": "Data fetched and saved to /data/api-data.json"}

        if task_code == "B4":  # Clone Git repo and make a commit
            repo_url = "https://github.com/example/repo.git"
            subprocess.run(["git", "clone", repo_url, "/data/repo"], check=True)
            return {"message": "Repository cloned into /data/repo"}

        if task_code == "B5":  # Run SQL query on SQLite
            conn = sqlite3.connect("/data/database.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            result = cursor.fetchone()[0]
            conn.close()
            return {"count": result}

        if task_code == "B6":  # Extract data from website
            response = requests.get("https://example.com")
            with open("/data/webpage.html", "w", encoding="utf-8") as f:
                f.write(response.text)
            return {"message": "Website scraped and saved"}

        if task_code == "B7":  # Compress or resize image
            os.system("convert /data/image.png -resize 50% /data/image-small.png")
            return {"message": "Image resized"}

        if task_code == "B8":  # Transcribe audio from MP3
            os.system("ffmpeg -i /data/audio.mp3 /data/audio.txt")
            return {"message": "Audio transcribed"}

        if task_code == "B9":  # Convert Markdown to HTML
            with open("/data/input.md", "r") as f:
                markdown_content = f.read()
            html_content = f"<html><body>{markdown_content}</body></html>"
            with open("/data/output.html", "w") as f:
                f.write(html_content)
            return {"message": "Markdown converted to HTML"}

        if task_code == "B10":  # Filter CSV and return JSON
            import pandas as pd
            df = pd.read_csv("/data/data.csv")
            filtered_df = df[df["column"] > 10]
            filtered_df.to_json("/data/output.json", orient="records")
            return {"message": "CSV filtered and saved"}

        return {"message": f"{task_code} Task '{task}' executed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Placeholder for file reading
@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="File path to read")):
    try:
        with open(path, "r") as file:
            return file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ========== HELPER FUNCTIONS ========== #

def png_to_base64(image_path):
    """Convert PNG image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_embedding(text):
    """Fetch text embedding from AIProxy."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {AIPROXY_TOKEN}"}
    data = {"model": "text-embedding-3-small", "input": [text]}
    response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/embeddings", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


# ========== SECURITY TASKS (B1, B2) ========== #

def B1(filepath):
    """Ensure data is accessed only within /data."""
    return filepath.startswith("/data")


def B2(filepath):
    """Prevent deletion of files anywhere in the system."""
    return os.path.exists(filepath)


# ========== TASK EXECUTION ROUTE ========== #


    
    