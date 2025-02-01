import os
import sys
from typing import List

import openai
from dotenv import load_dotenv
from openai import OpenAI
import json
import time

def get_path_from_project_root(relative_path):
    # Get the root directory of the project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '.'))

    # Get the absolute path of the file by joining the project root and the relative path
    file_path = os.path.join(project_root, relative_path)

    return file_path

def init_dotenv():
    # Get the path of the .env file
    if getattr(sys, 'frozen', False):
        # The application is running in a PyInstaller bundle
        env_path = os.path.join(sys._MEIPASS, '.env')
    else:
        # The application is running in a normal Python environment
        env_path = get_path_from_project_root('.env')
    load_dotenv(dotenv_path=env_path)

# Load the OpenAI API key from the .env file and initialize the OpenAI client
init_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
chatGptClient = OpenAI()
chatGptMessages = []

# Platforms:
_openai_ = "openai"
_ollama_ = "ollama"

# Chat models
gpt_3_5_turbo = "gpt-3.5-turbo"
gpt_3_5_turbo_instruct = "gpt-3.5-turbo-instruct"
gpt_4o_mini = "gpt-4o-mini"
gpt_4o = "gpt-4o"
o1 = "o1-preview"
llama3 = "llama3"

embedding_models = {
    _openai_: [
        gpt_3_5_turbo,
        gpt_3_5_turbo_instruct,
        gpt_4o_mini,
        gpt_4o,
        o1,
    ],
    _ollama_: [
        llama3,
    ]
}

current_chat_model = gpt_4o

system_prompt = ["You are a stock analyst"]

def get_user_prompt(stock_name: str) -> str:
    return f"""
    Please analyze the stocks {stock_name} using the following criteria. Afterwards, give a conclusion on how strong of a buy it is, and finally give it a rating from F- to S+. Format your response as a json with the fields [stock name, stock symbol, analysis, conclusion, dividend%, rating]. Please double check the json is formatted correctly with commas, braces, and quotes all in the right places.:
    Financial Health
    * Earnings Growth
    * Profit Margins
    * Debt & Cash Flow
    * Return on Capital (ROE, ROIC)
    Valuation & Stock Price
    * Price Relative to Highs/Lows
    * P/E, P/S, P/B, EV/EBITDA
    * Discounted Cash Flow (DCF)
    * Opportunism with current price point
    Market Sentiment & Leadership
    * Analyst Ratings
    * Institutional & Insider Activity
    * Leadership & Governance
    Risks & Competitive Positioning
    * Regulatory & Legal Issues
    * Macroeconomic & Industry Trends
    * Political landscape
    * Competitive Moat
    Growth & Stability
    * 5-Year Revenue & EPS Growth
    * Expansion & R&D Investment
    Stockholder Policies
    * Stock Buybacks
    Technical Analysis
    * RSI & Moving Averages
    """

def set_chat_model(model):
    global current_chat_model
    current_chat_model = model

def get_platform_of_model(model):
    for platform, models in embedding_models.items():
        if model in models:
            return platform
    return None

# Function to call the OpenAI API
def call_chat_agent(chatGptMessages):
    global current_chat_model
    platform = get_platform_of_model(current_chat_model)
    if platform == _openai_:
        completion = chatGptClient.chat.completions.create(
            model=current_chat_model,
            messages=chatGptMessages
        )
        response = completion.choices[0].message.content
        return response
    else:
        raise Exception(f"ChatGPT model {current_chat_model} not supported")

def call_chat_agent_with_context(prompt: str, rules: List[str]):

    full_context_for_chatagent = [
        {"role": "system", "content": "".join(rules)}
    ]

    user_message = {"role": "user", "content": prompt}
    full_context_for_chatagent += [user_message]

    # Call ChatGPT with the full context
    return call_chat_agent(full_context_for_chatagent)

def extract_response_json(response: str, stock: str):
    # Extract the json from the file (ignore everything before the first '{' and after the last '}')
    json_start = response.find('{')
    json_end = response.rfind('}')
    json_response = response[json_start:json_end+1]

    # Try and parse the json response
    try:
        response_dict = json.loads(json_response)
        # print(response_dict)
    except Exception as e:
        print(f"Error parsing response for stock {stock}: {e}\nResponse: {response}")

    # Get the "rating" field from the response (case insensitive).
    try:
        # Accept either "stock name" or "stock_name" as the stock name
        stock_name = response_dict.get("stock name", None)
        if stock_name is None:
            stock_name = response_dict.get("stock_name", None)
        stock_symbol = response_dict.get("stock symbol", None)
        if stock_symbol is None:
            stock_symbol = response_dict.get("stock symbol", None)
        stock_analysis = response_dict.get("analysis", None)
        stock_conclusion = response_dict.get("conclusion", None)
        stock_dividend = response_dict.get("dividend%", None)
        stock_rating = response_dict.get("rating", None)

        return {
            "stock name": stock_name,
            "stock symbol": stock_symbol,
            "analysis": stock_analysis,
            "conclusion": stock_conclusion,
            "dividend": stock_dividend,
            "rating": stock_rating
        }
    except Exception as e:
        print(f"Error parsing response for stock {stock}: {e}\nResponse: {response}")
        return {
            "stock name": stock,
            "stock symbol": stock,
            "analysis": None,
            "conclusion": None,
            "dividend": None,
            "rating": None
        }

def analyze_stock(stock: str):
    user_prompt = get_user_prompt(stock)
    try:
        response = call_chat_agent_with_context(user_prompt, system_prompt)
        
        # Parse the response
        return extract_response_json(response, stock)

    except Exception as e:
        print(f"Error processing stock {stock}: {e}\nResponse: {response}")


# Take in the context as a parameter
def main():
    # # Test line, comment out when running the full program
    # print(extract_response_json(temp, "ORCL"))
    # return

    # Start stopwatch
    start_time = time.time()

    # Load the stock names from the input file "input_stocks.json" (a list of stock symbols). The file is a txt file, line separated
    with open(get_path_from_project_root("input_stocks.txt"), "r") as f:
        stock_symbols = f.read().splitlines()

    print(f"\nBeginning stock analysis of {len(stock_symbols)} stocks\n")
    
    stock_analyses_long = []
    stock_analyses_short = []

    try:
        for i, stock_symbol in enumerate(stock_symbols):
            stock_analysis = analyze_stock(stock_symbol)
            stock_analyses_long.append(stock_analysis)

            # Format the short response
            stock_analysis_short = {
                "stock name": stock_analysis["stock name"],
                "rating": stock_analysis["rating"]
            }

            stock_analyses_short.append(stock_analysis_short)

            print(f"Finished analyzing {stock_symbol} ({i+1} of {len(stock_symbols)})")
            print(f"Rating: {stock_analysis['rating']}\n")
    except Exception as e:
        print(f"Error analyzing stock {stock_symbol}: {e}")
        
    try:
        # Sort the short responses by rating, from F- to S+ (must define the custom sort order)
        sort_order = ["S+", "S", "S-", "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F+", "F", "F-", None, "None"]
        stock_analyses_short.sort(key=lambda x: sort_order.index(x["rating"]))
        stock_analyses_long.sort(key=lambda x: sort_order.index(x["rating"]))

        # End stopwatch, convert time to int
        end_time = time.time()
        elapsed_time = int(end_time - start_time)
        print(f"Time to complete: {int(elapsed_time/60)}m{elapsed_time%60}s\n")
    except Exception as e:
        print(f"Error calculating elapsed time: {e}")

    # Write the formatted response to a file
    short_file_name = "output_short.json"
    long_file_name = "output_long.json"
    print(f"Writing output to files {short_file_name} and {long_file_name}\n")
    with open(get_path_from_project_root(short_file_name), "w") as f:
        json.dump(stock_analyses_short, f, indent=4)

    with open(get_path_from_project_root(long_file_name), "w") as f:
        json.dump(stock_analyses_long, f, indent=4)

if __name__ == "__main__":
    main()