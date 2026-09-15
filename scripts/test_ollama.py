import subprocess

prompt = "What is Artificial Intelligence?"

result = subprocess.run(
    ["ollama", "run", "qwen3:4b", prompt],
    capture_output=True,
    text=True
)

print(result.stdout)
