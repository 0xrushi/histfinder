from setuptools import setup, find_packages

setup(
    name="histfinder",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "groq",
        "python-dotenv",
        "langchain-chroma",
        "langchain-openai",
        "langchain-text-splitters",
        "pyautogui",
    ],
    entry_points={
        'console_scripts': [
            'histfinder=histfinder:run_async_main',
        ],
    },
    author="0xrushi",
    description="Search through command history using natural language",
    python_requires=">=3.11",
)
    
