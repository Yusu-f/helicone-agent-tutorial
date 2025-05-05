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

// LLM-based router function to determine query type and extract ticker if needed
async function routeQuery(query) {
  const response = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [
      {
        role: "system",
        content: `You are a financial query router that determines whether a query requires real-time stock data or a financial term definition.
        
        If the query is about a specific stock's price, performance, or news, categorize it as "STOCK_DATA" and extract the ticker symbol.
        If the query is about a financial term, concept, or definition, categorize it as "FINANCIAL_TERM".
        
        Respond with a JSON object with two fields:
        - routeTo: Either "STOCK_DATA" or "FINANCIAL_TERM"
        - ticker: The ticker symbol if routeTo is "STOCK_DATA" (omit this field otherwise)
        `
      },
      {
        role: "user",
        content: query
      }
    ],
    temperature: 0.1,
    response_format: { type: "json_object" }
  });
  
  // Parse the JSON response
  try {
    const result = JSON.parse(response.choices[0].message.content || "{}");
    return {
      routeTo: result.routeTo,
      ticker: result.ticker
    };
  } catch (error) {
    console.error("Error parsing router response:", error);
    // Default to financial term if parsing fails
    return { routeTo: "FINANCIAL_TERM" };
  }
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

// Main function to process user queries
async function processQuery(query, vectorStore) {
  // Use LLM to route the query
  console.log("Routing query...");
  const routingDecision = await routeQuery(query);
  console.log(`Query routed to: ${routingDecision.routeTo}${routingDecision.ticker ? `, Ticker: ${routingDecision.ticker}` : ''}`);
  
  // Handle based on routing decision
  if (routingDecision.routeTo === "STOCK_DATA" && routingDecision.ticker) {
    // Get stock data and news
    console.log(`Fetching data for ${routingDecision.ticker}...`);
    const stockData = await getStockData(routingDecision.ticker);
    const newsData = await getStockNews(routingDecision.ticker);
    
    // Create messages array for OpenAI
    const messages = [
      {
        role: "system",
        content: `You are a financial research assistant that provides accurate, helpful information about stocks and financial markets. 
        When discussing stock data, clearly present the price, change, and other metrics in a readable format. 
        When presenting news, provide brief summaries with sources. 
        Always include appropriate disclaimers about investment risks.`
      },
      // Add chat history
      ...chatHistory,
      // Add current query
      {
        role: "user",
        content: `User query: ${query}\n\nStock Data: ${JSON.stringify(stockData)}\n\nNews Data: ${JSON.stringify(newsData)}`
      }
    ];
    
    // Generate response using OpenAI with Helicone monitoring
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages,
      temperature: 0.1,
    });
    
    return response.choices[0].message.content;
  } else {
    // This is a financial term query - use RAG
    console.log("Searching for financial term definitions...");
    
    // Get relevant documents from vector store
    const results = await vectorStore.similaritySearch(query, 2);
    
    // Add Helicone property for relevant document count
    const hasRelevantResults = results.length > 0;
    
    // Check if we have relevant results
    if (!hasRelevantResults || results[0].score < 0.7) {
      return "I don't have specific information about this financial term or concept in my knowledge base. For reliable financial information, please consider consulting a financial advisor, visiting financial education websites like Investopedia, or checking resources from financial regulatory bodies.";
    }
    
    // Create messages array for OpenAI
    const messages = [
      {
        role: "system",
        content: `You are a financial research assistant that provides accurate definitions of financial terms and concepts. 
        Use ONLY the provided context to answer questions, without adding any information not contained in the context.
        If the provided context doesn't fully address the user's question, acknowledge the limitations of your response.
        Keep your responses clear, concise, and educational.`
      },
      // Add chat history
      ...chatHistory,
      // Add current query
      {
        role: "user",
        content: `User query: ${query}\n\nRelevant information:\n${results.map(doc => doc.pageContent).join('\n\n')}`
      }
    ];
    
    // Generate response using OpenAI with RAG context and Helicone monitoring
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages,
      temperature: 0.1,
    });
    
    return response.choices[0].message.content;
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