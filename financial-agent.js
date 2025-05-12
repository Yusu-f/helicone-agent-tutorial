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

// Sample company profiles for RAG - fake financial data
const companyProfiles = [
  `
  # TechVision Inc. (TVIX)
  
  Industry: Technology
  Founded: 2005
  Headquarters: San Francisco, CA
  
  TechVision is a leading AI and machine learning company specializing in computer vision solutions. Their flagship product, VisionCore, is used by major automotive manufacturers for autonomous driving systems.
  
  Recent developments:
  - Announced partnership with AutoDrive to enhance autonomous vehicle safety features
  - Introduced new AI chip with 40% better performance than previous generation
  - Expanding into healthcare imaging with acquisition of MedSight Technologies
  
  Financial highlights:
  - Annual revenue: $3.2B (up 18% YoY)
  - Profit margin: 22%
  - R&D spending: $780M (24% of revenue)
  `,
  `
  # GreenEnergy Corp (GRNE)
  
  Industry: Renewable Energy
  Founded: 2010
  Headquarters: Austin, TX
  
  GreenEnergy specializes in solar and wind energy solutions with a focus on energy storage technology. Their battery systems are used in both residential and commercial applications.
  
  Recent developments:
  - Launched next-generation home battery with 30% increased capacity
  - Secured $500M contract to build solar farm in Nevada
  - Expanding manufacturing facilities in Texas and Arizona
  
  Financial highlights:
  - Annual revenue: $1.8B (up 25% YoY)
  - Profit margin: 14%
  - Net cash position: $620M
  `,
  `
  # HealthPlus Inc. (HLTH)
  
  Industry: Healthcare
  Founded: 1998
  Headquarters: Boston, MA
  
  HealthPlus develops innovative medical devices and digital health platforms. Their diabetes management system has captured significant market share in the US.
  
  Recent developments:
  - FDA approval for next-generation continuous glucose monitor
  - Expanded telemedicine platform to include mental health services
  - Strategic partnership with major insurance providers
  
  Financial highlights:
  - Annual revenue: $2.4B (up 12% YoY)
  - Profit margin: 18%
  - International sales: 35% of revenue
  `,
  `
  # DigitalFinance Group (DFG)
  
  Industry: Fintech
  Founded: 2015
  Headquarters: New York, NY
  
  DigitalFinance provides blockchain-based payment solutions and digital banking services to both consumers and businesses.
  
  Recent developments:
  - Launched small business lending platform with AI-powered risk assessment
  - Obtained banking license in European Union
  - Integrated with major e-commerce platforms
  
  Financial highlights:
  - Annual revenue: $950M (up 40% YoY)
  - Profit margin: 8%
  - User base: 12 million (up 30% YoY)
  `,
  `
  # ConsumerBrands Corp (CNBC)
  
  Industry: Consumer Goods
  Founded: 1975
  Headquarters: Chicago, IL
  
  ConsumerBrands manages a portfolio of household products, personal care items, and food brands with strong presence in North America and Europe.
  
  Recent developments:
  - Sustainability initiative to make all packaging recyclable by 2026
  - Expansion into Asian markets
  - Divested underperforming snack food division
  
  Financial highlights:
  - Annual revenue: $8.5B (up 5% YoY)
  - Profit margin: 15%
  - Dividend yield: 3.2%
  `
];

// Function to initialize the vector store with company profiles
async function initializeVectorStore() {
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });
  
  const docs = companyProfiles.map(
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

// Function to search for company information
async function searchCompanyInfo(query, vectorStore) {
  console.log("Searching for company information in knowledge base...");
  
  // Get relevant documents from vector store with similarity scores
  const resultsWithScores = await vectorStore.similaritySearchWithScore(query, 2);
    
  // Check if we have relevant results with good similarity scores
  if (resultsWithScores.length === 0 || resultsWithScores[0][1] < 0.9) {
    console.log("No relevant company information found in the knowledge base.");
    return {
      found: false,
      message: "No relevant company information found in the knowledge base."
    };
  }
  
  // Extract just the documents from the results for context
  const relevantDocs = resultsWithScores.map(([doc]) => doc);
  
  return {
    found: true,
    documents: relevantDocs
  };
}

// Define the tools for our agent
const tools = [
  {
    type: "function",
    function: {
      name: "getStockData",
      description: "Get current price and other market information for a specific stock by ticker symbol",
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
      name: "searchCompanyInfo",
      description: "Search for detailed company information in the knowledge base",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The company name or topic to search for"
          }
        },
        required: ["query"]
      }
    }
  }
];

// Simple helper to map tool calls to functions
async function callHelper(name, args, vectorStore) {
  switch(name) {
    case "getStockData":
      return await getStockData(args.ticker);
    case "getStockNews":
      return await getStockNews(args.ticker);
    case "searchCompanyInfo":
      return await searchCompanyInfo(args.query, vectorStore);
    default:
      return { error: `Unknown tool: ${name}` };
  }
}

// Tiny agent loop for processing queries (≤ 20 lines)
async function processQuery(userQuery, vectorStore) {
  let messages = [
    {
      role: "system",
      content: `You're a financial assistant. Use tools when needed. If you have enough information to answer, reply normally.`
    },
    { role: "user", content: userQuery }
  ];
  
  // Add chat history for context if available
  if (chatHistory.length > 0) {
    messages.splice(1, 0, ...chatHistory);
  }
  
  while (true) {
    console.log("Sending query to OpenAI...");
    const llmResp = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      tools,
      messages,
      temperature: 0.1,
    });
    
    const msg = llmResp.choices[0].message;
    
    if (msg.tool_calls && msg.tool_calls.length > 0) {
      // Execute the helper
      const toolCall = msg.tool_calls[0];
      const functionName = toolCall.function.name;
      const functionArgs = JSON.parse(toolCall.function.arguments);
      
      console.log(`Executing ${functionName} with args:`, functionArgs);
      const toolResult = await callHelper(functionName, functionArgs, vectorStore);
      
      // Push feedback & loop
      messages.push(msg);  // LLM's tool call
      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        name: functionName,
        content: JSON.stringify(toolResult)
      });
      continue;
    }
    
    // No tool call → LLM has produced the final answer
    return msg.content;
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
  console.log("Ask about stock prices, news, or company information.");
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
        
        // Process query with the agentic approach
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