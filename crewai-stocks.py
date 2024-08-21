# IMPORT DAS LIBS
import json
import os
import traceback
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# CRIANDO YAHOO FINANCE TOOL
def fetch_stock_price(ticket):
    try:
        st.write(f"Fetching stock price for ticket: {ticket}")
        start_date = datetime.now() - timedelta(days=365)  # Use data dos últimos 12 meses
        end_date = datetime.now()
        stock = yf.download(ticket, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        if stock.empty:
            st.warning(f"No data found for ticket: {ticket}")
            return None
        else:
            st.write(f"Data fetched for {ticket}: {stock.head()}")
            return stock
    except Exception as e:
        st.error(f"Error fetching stock price: {e}")
        return None

yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetches stock prices for {ticket} from the last year about a specific company from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket)
)

# IMPORTANDO OPENAI LLM - GPT
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")

# AGENT: Stock Price Analyst
stockPriceAnalyst = Agent(
    role="Senior stock price Analyst",
    goal="Find the {ticket} stock price and analyze trends",
    backstory="""You're highly experienced in analyzing the price of a specific stock and making predictions about its future price.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_delegation=False
)

# TASK: Get Stock Price
getStockPrice = Task(
    description="Analyze the stock {ticket} price history and create a trend analysis of up, down, or sideways",
    expected_output="""Specify the current trend stock price - up, down, or sideways.
    eg. stock= 'AAPL, price UP'""",
    agent=stockPriceAnalyst
)

# IMPORTANDO A TOOL DE SEARCH
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

# AGENT: News Analyst
newsAnalyst = Agent(
    role="Stock News Analyst",
    goal="""Create a short summary of the market news related to the stock {ticket} company. Specify the current trend - up, down, or sideways with the news context.
    For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory="""You're highly experienced in analyzing market trends and news and have tracked assets for more than 10 years.
    You're also a master-level analyst in the traditional markets and have a deep understanding of human psychology.
    You understand news, their titles, and information but look at them with a healthy dose of skepticism.
    You also consider the source of the news articles.""",
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    tools=[search_tool],
    allow_delegation=False
)

# TASK: Get News
get_news = Task(
    description=f"""Take the stock and always include BTC to it (if not requested).
    Use the search tool to search each one individually.
    The current date is {datetime.now().strftime("%Y-%m-%d")}.
    Compose the results into a helpful report""",
    expected_output="""A summary of the overall market and one-sentence summary for each requested asset.
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>""",
    agent=newsAnalyst
)

# AGENT: Stock Analyst Writer
stockAnalystWrite = Agent(
    role="Senior Stock Analyst Writer",
    goal="""Analyze the trends in price and news and write an insightful, compelling, and informative 3-paragraph long newsletter based on the stock report and price trend.""",
    backstory="""You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories and narratives that resonate with wider audiences.
    You understand macro factors and combine multiple theories - e.g., cycle theory and fundamental analysis.
    You're able to hold multiple opinions when analyzing anything.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True
)

# TASK: Write Analysis
writeAnalyses = Task(
    description="""Use the stock price trend and stock news report to create an analysis and write the newsletter about the {ticket} company
    that is brief and highlights the most important points. Focus on the stock price trend, news, and fear/greed score. What are near future considerations?
    Include the previous analysis of stock trend and news summary.""",
    expected_output="""An eloquent 3-paragraph newsletter formatted as markdown in an easy-to-read manner. It should contain:
    - 3 bullets executive summary
    - Introduction - set the overall picture and spike up the interest
    - Main part provides the meat of the analysis including the news summary and fear/greed scores
    - Summary - key facts and concrete future trend prediction - up, down, or sideways.""",
    agent=stockAnalystWrite,
    context=[getStockPrice, get_news]
)

# CRIANDO A EQUIPE
crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks=[getStockPrice, get_news, writeAnalyses],
    verbose=True,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

# CONFIGURANDO STREAMLIT
with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label="Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        st.write("Fetching results for ticket:", topic)
        try:
            # Debug: Verificar entradas antes do kickoff
            st.write("Starting kickoff with input:", {'ticket': topic})
            results = crew.kickoff(inputs={'ticket': topic})
            st.write("Raw results:", results)  # Log de resultados brutos
            final_output = results.get('final_output', 'No final output found')
            st.subheader("Results of your research:")
            st.write(final_output)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write(f"Full traceback: {traceback.format_exc()}")
