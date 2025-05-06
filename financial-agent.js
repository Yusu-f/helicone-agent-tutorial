import dotenv from 'dotenv';
import axios from 'axios';
import { OpenAI } from 'openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Document } from '@langchain/core/documents';
import readline from 'readline';

// Load environment variables
dotenv.config();

// Initialize OpenAI client with Helicone monitoring
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "https://oai.helicone.ai/v1",
  defaultHeaders: {
    "Helicone-Auth": `Bearer ${process.env.HELICONE_API_KEY}`,
  },
});

// Alpha Vantage API key
const ALPHA_VANTAGE_API_KEY = process.env.ALPHA_VANTAGE_API_KEY;

// Initialize chat history
const chatHistory = [];

// Sample financial terms for RAG - definitions of common finance terms
const financialTerms = [
  `
  # P/E Ratio (Price-to-Earnings Ratio)
  
  A valuation ratio calculated by dividing a company's current share price by its earnings per share (EPS).
  
  A high P/E ratio may indicate that investors expect higher growth in the future compared to companies with a lower P/E.
  However, a high P/E ratio might also suggest that a stock is overvalued.
  
  The average P/E ratio varies by industry and market conditions. As of 2023, the average P/E ratio for the S&P 500 is approximately 20-25.
  `,
  `
  # Market Capitalization
  
  The total dollar market value of a company's outstanding shares of stock. Calculated by multiplying the total number of a company's outstanding shares by the current market price of one share.
  
  Market cap classifications:
  - Mega-cap: $200 billion+
  - Large-cap: $10-200 billion
  - Mid-cap: $2-10 billion
  - Small-cap: $300 million-$2 billion
  - Micro-cap: $50-300 million
  - Nano-cap: Below $50 million
  `,
  `
  # Dividend Yield
  
  A financial ratio that shows how much a company pays out in dividends each year relative to its stock price.
  
  Calculated as: Annual Dividends per Share / Price per Share
  
  High dividend yields may indicate:
  - The stock is undervalued
  - The company is mature and not reinvesting as much in growth
  - In some cases, unsustainable dividend payments
  
  Sectors known for higher dividend yields include utilities, telecommunications, and some consumer staples.
  `,
  `
  # Bull Market
  
  A financial market condition where prices are rising or expected to rise. Typically characterized by:
  
  - Investor confidence and optimism
  - Economic strength
  - Rising stock prices, typically at least 20% from recent lows
  - Increased trading volume and market participation
  
  The longest bull market in U.S. history lasted from 2009 to 2020, following the financial crisis.
  `,
  `
  # Bear Market
  
  A financial market condition where prices are falling or expected to fall. Typically characterized by:
  
  - Investor pessimism and negative sentiment
  - Economic weakness or uncertainty
  - Falling stock prices, typically at least 20% from recent highs
  - Reduced trading volume and market participation
  
  Bear markets often precede economic recessions but not always.
  `,
  `
  # Bonds
  
  Debt securities where an investor loans money to an entity (corporate or governmental) that borrows the funds for a defined period of time at a fixed interest rate.
  
  Key characteristics:
  - Face/par value: The amount paid to the bondholder at maturity
  - Coupon rate: The interest rate paid on the face value
  - Maturity date: When the bond issuer returns the principal to investors
  - Yield: The total return anticipated on a bond if held until maturity
  
  Types include Treasury bonds, municipal bonds, corporate bonds, and junk bonds.
  `,
  `
  # ETF (Exchange-Traded Fund)
  
  An investment fund traded on stock exchanges, holding assets such as stocks, bonds, or commodities. ETFs typically track an index, sector, commodity, or other asset.
  
  Key features:
  - Trade like stocks on exchanges throughout the day
  - Usually have lower expense ratios than mutual funds
  - Often more tax efficient than mutual funds
  - Provide diversification similar to mutual funds
  - Most ETFs are passively managed to track an index
  
  Popular examples include SPY (S&P 500 ETF), QQQ (Nasdaq-100 ETF), and VTI (Vanguard Total Stock Market ETF).
  `,
  `
  # Mutual Fund
  
  An investment vehicle made up of a pool of money collected from many investors to invest in securities such as stocks, bonds, and other assets.
  
  Key features:
  - Professionally managed by fund managers
  - Priced once per day after market close (NAV)
  - May have minimum investment requirements
  - Available in active or passive management styles
  - Can have higher expense ratios than ETFs
  
  Types include stock funds, bond funds, money market funds, target-date funds, and balanced/hybrid funds.
  `,
  `
  # EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization)
  
  A measure of a company's overall financial performance, used as an alternative to net income in some circumstances.
  
  EBITDA = Revenue - Expenses (excluding interest, taxes, depreciation, and amortization)
  
  EBITDA can provide a clearer picture of operating performance by eliminating the effects of financing and accounting decisions. However, it does not account for capital expenditures, which can be significant for some businesses.
  `,
  `
  # IPO (Initial Public Offering)
  
  The process of offering shares of a private corporation to the public in a new stock issuance for the first time.
  
  Key aspects:
  - Allows companies to raise capital from public investors
  - Typically involves investment banks as underwriters
  - Requires extensive financial disclosure and regulatory filings
  - Share price is determined through underwriter's due diligence and market demand
  - Often includes a "quiet period" before and after the offering
  
  Notable recent IPOs include Airbnb, Coinbase, and Rivian.
  `,
  `
  # ROI (Return on Investment)
  
  A performance measure used to evaluate the efficiency or profitability of an investment relative to its cost.
  
  ROI = (Net Profit / Cost of Investment) × 100%
  
  A higher ROI indicates a more efficient investment. However, ROI does not account for time held or risk involved, making it somewhat limited for comparing investments.
  `,
  `
  # Liquidity
  
  The ease with which an asset or security can be converted into cash without affecting its market price.
  
  High liquidity assets:
  - Cash and cash equivalents
  - Major currencies
  - Blue-chip stocks
  - Treasury bills
  
  Low liquidity assets:
  - Real estate
  - Private company shares
  - Collectibles
  - Thinly traded securities
  
  Liquidity risk refers to the possibility that an investor may not be able to buy or sell an investment quickly enough in the market to prevent a loss.
  `,
  `
  # Hedge Fund
  
  A pooled investment fund that uses various complex strategies to earn active returns for its investors. Typically only available to accredited or institutional investors.
  
  Common strategies:
  - Long/short equity
  - Market neutral
  - Global macro
  - Event-driven
  - Quantitative
  
  Hedge funds often use leverage, derivatives, and short positions, which can increase both potential returns and risks.
  `,
  `
  # Basis Point
  
  A unit of measure used in finance to describe the percentage change in the value of financial instruments or the rate change in an index or other benchmark.
  
  One basis point is equivalent to 0.01% (1/100th of a percent) or 0.0001 in decimal form.
  
  For example:
  - An increase from 5.00% to 5.25% is a rise of 25 basis points
  - A decrease from 3.5% to 3.0% is a drop of 50 basis points
  
  Basis points are commonly used when discussing interest rates, yields, and changes in indexes.
  `
];

// Function to initialize the vector store with financial term definitions
async function initializeVectorStore() {
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });
  
  const docs = financialTerms.map(
    (text) => new Document({ pageContent: text })
  );
  
  return MemoryVectorStore.fromDocuments(docs, embeddings);
}

// Function to get stock price data from Alpha Vantage
async function getStockData(ticker) {
  try {
    const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${ticker}&apikey=${ALPHA_VANTAGE_API_KEY}`;
    const response = await axios.get(url);
    
    if (response.data['Global Quote'] && Object.keys(response.data['Global Quote']).length > 0) {
      const quote = response.data['Global Quote'];
      return {
        symbol: ticker.toUpperCase(),
        price: parseFloat(quote['05. price']),
        change: parseFloat(quote['09. change']),
        changePercent: quote['10. change percent'],
        volume: parseInt(quote['06. volume']),
        latestTradingDay: quote['07. latest trading day']
      };
    } else {
      return { error: `No data found for ticker ${ticker}` };
    }
  } catch (error) {
    console.error('Error fetching stock data:', error);
    return { error: `Failed to get stock data for ${ticker}` };
  }
}

// Function to get latest news for a ticker
async function getStockNews(ticker) {
  try {
    const url = `https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=${ticker}&apikey=${ALPHA_VANTAGE_API_KEY}`;
    const response = await axios.get(url);
    
    if (response.data.feed && response.data.feed.length > 0) {
      return response.data.feed.slice(0, 3).map((item) => ({
        title: item.title,
        summary: item.summary,
        source: item.source,
        url: item.url,
        timePublished: item.time_published
      }));
    } else {
      return { error: `No news found for ticker ${ticker}` };
    }
  } catch (error) {
    console.error('Error fetching news:', error);
    return { error: `Failed to get news for ${ticker}` };
  }
}

// Function to perform vector search for financial terms
async function searchFinancialTerms(query, vectorStore) {
  console.log("Searching for financial term definitions in knowledge base...");
  
  // Get relevant documents from vector store with similarity scores
  const resultsWithScores = await vectorStore.similaritySearchWithScore(query, 2);
    
  // Check if we have relevant results with good similarity scores
  if (resultsWithScores.length === 0 || resultsWithScores[0][1] < 0.7) {
    console.log("No relevant financial terms found in the knowledge base.");
    return { 
      found: false,
      message: "No relevant financial terms found in the knowledge base."
    };
  }
  
  // Extract just the documents from the results for context
  const relevantDocs = resultsWithScores.map(([doc]) => doc);
  
  return {
    found: true,
    documents: relevantDocs
  };
}

// Main function to process user queries
async function processQuery(query, vectorStore) {
  // Define OpenAI tools for function calling
  const tools = [
    {
      type: "function",
      function: {
        name: "getStockData",
        description: "Get current price and other information for a specific stock by ticker symbol",
        parameters: {
          type: "object",
          properties: {
            ticker: {
              type: "string",
              description: "The stock ticker symbol, e.g., AAPL for Apple Inc."
            }
          },
          required: ["ticker"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "getStockNews",
        description: "Get the latest news articles for a specific stock by ticker symbol",
        parameters: {
          type: "object",
          properties: {
            ticker: {
              type: "string",
              description: "The stock ticker symbol, e.g., AAPL for Apple Inc."
            }
          },
          required: ["ticker"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "searchFinancialTerms",
        description: "Search for definitions of financial terms and concepts in the knowledge base",
        parameters: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "The financial term or concept to search for"
            }
          },
          required: ["query"]
        }
      }
    }
  ];
  
  // Create messages array with system prompt and chat history
  const messages = [
    {
      role: "system",
      content: `You are a financial research assistant that provides accurate information about stocks and financial concepts.
      
      When a user asks about a specific stock, use the getStockData and getStockNews functions to retrieve current information.
      When a user asks about a financial term or concept, use the searchFinancialTerms function to find relevant definitions.
      
      Present information clearly, with appropriate formatting. For stock data, include price, change, and other key metrics.
      For news, provide brief summaries with sources. For financial terms, give clear explanations based on the provided definitions.
      
      Always include appropriate disclaimers about investment risks when discussing stocks.`
    },
    // Add chat history
    ...chatHistory,
    // Add current query
    {
      role: "user",
      content: query
    }
  ];
  
  // Get response from LLM on how to process query
  console.log("Sending query to OpenAI...");
  const initialResponse = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages,
    tools,
    temperature: 0.1,
  });
  
  const initialMessage = initialResponse.choices[0].message;
  
  // Check if the model wants to call a function
  if (initialMessage.tool_calls) {    
    // Create an array to store messages for this interaction
    const messageHistory = [...messages, initialMessage];
    
    // Process each tool call
    for (const toolCall of initialMessage.tool_calls) {
      const functionName = toolCall.function.name;
      const functionArgs = JSON.parse(toolCall.function.arguments);
      
      let functionResponse;
      
      // Execute the appropriate function based on the call
      if (functionName === "getStockData") {
        console.log(`Fetching stock data for ${functionArgs.ticker}...`);
        functionResponse = await getStockData(functionArgs.ticker);
      } else if (functionName === "getStockNews") {
        console.log(`Fetching news for ${functionArgs.ticker}...`);
        functionResponse = await getStockNews(functionArgs.ticker);
      } else if (functionName === "searchFinancialTerms") {
        console.log(`Searching for financial terms matching: ${functionArgs.query}`);
        const searchResults = await searchFinancialTerms(functionArgs.query, vectorStore);
        
        if (searchResults.found) {
          functionResponse = {
            found: true,
            definitions: searchResults.documents.map(doc => doc.pageContent)
          };
        } else {
          functionResponse = {
            found: false,
            message: searchResults.message
          };
        }
      }      
      
      // Add the function response to the message history
      messageHistory.push({
        role: "tool",
        tool_call_id: toolCall.id,
        name: functionName,
        content: JSON.stringify(functionResponse)
      });
    }
    
    // Get a final response from the model with the function results
    console.log("Getting final response with tool results...");
    const finalResponse = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: messageHistory,
      temperature: 0.1,
    });
    
    return finalResponse.choices[0].message.content;
  } else {
    // If no function was called, return the initial response
    return initialMessage.content;
  }
}

// Main function
async function main() {
  console.log("Initializing Financial Research Assistant...");
  
  // Check for API keys
  if (!process.env.OPENAI_API_KEY) {
    console.error("Error: OPENAI_API_KEY not found in environment variables");
    process.exit(1);
  }
  
  if (!process.env.ALPHA_VANTAGE_API_KEY) {
    console.error("Error: ALPHA_VANTAGE_API_KEY not found in environment variables");
    process.exit(1);
  }
  
  if (!process.env.HELICONE_API_KEY) {
    console.error("Error: HELICONE_API_KEY not found in environment variables");
    process.exit(1);
  }
  
  // Initialize vector store
  const vectorStore = await initializeVectorStore();
  
  // Create readline interface
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  
  console.log("\n===== Financial Research Assistant =====");
  console.log("Ask about stock prices, news, or financial terms.");
  console.log("Type 'exit' to quit.");
  console.log("=======================================\n");
  
  // Start conversation loop
  const askQuestion = () => {
    rl.question("\nWhat would you like to know? ", async (query) => {
      if (query.toLowerCase() === "exit") {
        console.log("Thank you for using the Financial Research Assistant. Goodbye!");
        rl.close();
        return;
      }
      
      try {
        console.log("\nResearching your question...");
        
        // Process query
        const answer = await processQuery(query, vectorStore);
        
        // Display answer
        console.log("\nAnswer:", answer);
        
        // Update chat history (maintain a limited history)
        chatHistory.push({ role: "user", content: query });
        chatHistory.push({ role: "assistant", content: answer || "" });
        
        // Keep only the last 4 exchanges (8 messages)
        if (chatHistory.length > 8) {
          chatHistory.splice(0, 2);
        }
        
        // Continue conversation
        askQuestion();
      } catch (error) {
        console.error("Error processing query:", error);
        console.log("\nSorry, I encountered an error while researching your question. Please try again.");
        askQuestion();
      }
    });
  };
  
  // Start the conversation
  askQuestion();
}

// Run the assistant
main().catch(console.error);