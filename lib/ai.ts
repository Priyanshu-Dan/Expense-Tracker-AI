import { GoogleGenAI } from '@google/genai';

interface RawInsight {
  type?: string;
  title?: string;
  message?: string;
  action?: string;
  confidence?: number;
}

const client = new GoogleGenAI({});

export interface ExpenseRecord {
  id: string;
  amount: number;
  category: string;
  description: string;
  date: string;
}

export interface AIInsight {
  id: string;
  type: 'warning' | 'info' | 'success' | 'tip';
  title: string;
  message: string;
  action?: string;
  confidence: number;
}

// In-memory cache to reduce API calls
interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

// Cache insights for 15 minutes — no need to re-analyze on every page load
const INSIGHTS_CACHE_TTL = 15 * 60 * 1000; // 15 minutes
let insightsCache: CacheEntry<AIInsight[]> | null = null;

// Cache category suggestions — same description always gets the same category
const categoryCache = new Map<string, string>();
const MAX_CATEGORY_CACHE_SIZE = 100;

// Cache AI answers — same question doesn't need to be asked twice
const answerCache = new Map<string, string>();
const MAX_ANSWER_CACHE_SIZE = 50;


export async function generateExpenseInsights(
  expenses: ExpenseRecord[]
): Promise<AIInsight[]> {
  try {
    // Check cache first — return cached insights if still fresh
    if (insightsCache && Date.now() - insightsCache.timestamp < INSIGHTS_CACHE_TTL) {
      console.log('Returning cached AI insights');
      return insightsCache.data;
    }

    // Prepare expense data for AI analysis
    const expensesSummary = expenses.map((expense) => ({
      amount: expense.amount,
      category: expense.category,
      description: expense.description,
      date: expense.date,
    }));

    const prompt = `Analyze the following expense data and provide 3-4 actionable financial insights. 
    Return a JSON array of insights with this structure:
    {
      "type": "warning|info|success|tip",
      "title": "Brief title",
      "message": "Detailed insight message with specific numbers when possible",
      "action": "Actionable suggestion",
      "confidence": 0.8
    }

    Expense Data:
    ${JSON.stringify(expensesSummary, null, 2)}

    Focus on:
    1. Spending patterns (day of week, categories)
    2. Budget alerts (high spending areas)
    3. Money-saving opportunities
    4. Positive reinforcement for good habits

    Return only valid JSON array, no additional text.`;

    let retries = 2;
    let responseText = '';

    while (retries > 0) {
      try {
        const interaction = await client.interactions.create({
          model: 'Gemini 3.5 Flash Lite',
          system_instruction:
            'You are a financial advisor AI that analyzes spending patterns and provides actionable insights. Always respond with valid JSON only.',
          input: prompt,
        });

        responseText = interaction.output_text || '';
        if (responseText.trim()) {
          break;
        }
        
        throw new Error('Empty response content from AI');
      } catch (err: unknown) {
        retries--;
        
        if (retries > 0) {
          await new Promise(resolve => setTimeout(resolve, 2000));
        } else {
          if (err instanceof Error && err.message === 'Empty response content from AI') {
            throw new Error('No response from AI after multiple attempts.');
          }
          throw err;
        }
      }
    }

    if (!responseText) {
      throw new Error('No response from AI');
    }

    // Clean the response by extracting the JSON array
    let cleanedResponse = responseText.trim();
    
    const startIdx = cleanedResponse.indexOf('[');
    const endIdx = cleanedResponse.lastIndexOf(']');
    
    if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
      cleanedResponse = cleanedResponse.substring(startIdx, endIdx + 1);
    } else {
      throw new Error(`AI returned invalid format. Received: ${cleanedResponse.substring(0, 50)}...`);
    }

    let insights;
    try {
      insights = JSON.parse(cleanedResponse);
    } catch (parseError) {
      console.error('Failed to parse JSON:', cleanedResponse);
      throw new Error('AI returned malformed JSON data.');
    }

    if (!Array.isArray(insights)) {
      throw new Error('AI did not return an array of insights.');
    }

    const formattedInsights = insights.map(
      (insight: RawInsight, index: number) => ({
        id: `ai-${Date.now()}-${index}`,
        type: (insight.type as 'warning' | 'info' | 'success' | 'tip') || 'info',
        title: insight.title || 'AI Insight',
        message: insight.message || 'Analysis complete',
        action: insight.action,
        confidence: insight.confidence || 0.8,
      })
    );

    // Store in cache
    insightsCache = {
      data: formattedInsights,
      timestamp: Date.now(),
    };

    return formattedInsights;
  } catch (error) {
    console.error('❌ Error generating AI insights:', error);

    return [
      {
        id: 'fallback-1',
        type: 'warning',
        title: 'AI Analysis Failed',
        message: `Reason: ${error instanceof Error ? error.message : 'Unknown error'}`,
        action: 'Please let the AI assistant know this exact error',
        confidence: 0,
      },
    ];
  }
}

export async function categorizeExpense(description: string): Promise<string> {
  try {
    // Normalize the description for cache lookup
    const normalizedDesc = description.trim().toLowerCase();

    // Check cache first
    if (categoryCache.has(normalizedDesc)) {
      console.log('Returning cached category for:', normalizedDesc);
      return categoryCache.get(normalizedDesc)!;
    }

    const interaction = await client.interactions.create({
      model: 'gemma-4-26b',
      system_instruction:
        'You are an expense categorization AI. Categorize expenses into one of these categories: Food, Transportation, Entertainment, Shopping, Bills, Healthcare, Other. Respond with only the category name.',
      input: `Categorize this expense: "${description}"`,
    });

    const category = interaction.output_text?.trim();

    const validCategories = [
      'Food',
      'Transportation',
      'Entertainment',
      'Shopping',
      'Bills',
      'Healthcare',
      'Other',
    ];

    const finalCategory = validCategories.includes(category || '')
      ? category!
      : 'Other';

    // Store in cache (evict oldest if full)
    if (categoryCache.size >= MAX_CATEGORY_CACHE_SIZE) {
      const firstKey = categoryCache.keys().next().value;
      if (firstKey) categoryCache.delete(firstKey);
    }
    categoryCache.set(normalizedDesc, finalCategory);

    return finalCategory;
  } catch (error) {
    console.error('❌ Error categorizing expense:', error);
    return 'Other';
  }
}

export async function generateAIAnswer(
  question: string,
  context: ExpenseRecord[]
): Promise<string> {
  try {
    // Check cache first
    const cacheKey = question.trim().toLowerCase();
    if (answerCache.has(cacheKey)) {
      console.log('Returning cached AI answer for:', cacheKey);
      return answerCache.get(cacheKey)!;
    }

    const expensesSummary = context.map((expense) => ({
      amount: expense.amount,
      category: expense.category,
      description: expense.description,
      date: expense.date,
    }));

    const prompt = `Based on the following expense data, provide a detailed and actionable answer to this question: "${question}"

    Expense Data:
    ${JSON.stringify(expensesSummary, null, 2)}

    Provide a comprehensive answer that:
    1. Addresses the specific question directly
    2. Uses concrete data from the expenses when possible
    3. Offers actionable advice
    4. Keeps the response concise but informative (2-3 sentences)
    
    Return only the answer text, no additional formatting.`;

    let retries = 2;
    let responseText = '';

    while (retries > 0) {
      try {
        const interaction = await client.interactions.create({
          model: 'Gemini 3.5 Flash Lite',
          system_instruction:
            'You are a helpful financial advisor AI that provides specific, actionable answers based on expense data. Be concise but thorough.',
          input: prompt,
        });

        responseText = interaction.output_text || '';
        if (responseText.trim()) {
          break;
        }

        throw new Error('Empty response content from AI');
      } catch (err: unknown) {
        retries--;
        
        if (retries > 0) {
          await new Promise(resolve => setTimeout(resolve, 2000));
        } else {
          if (err instanceof Error && err.message === 'Empty response content from AI') {
            throw new Error('No response from AI after multiple attempts.');
          }
          throw err;
        }
      }
    }

    if (!responseText) {
      throw new Error('No response from AI');
    }

    const trimmedResponse = responseText.trim();

    // Store in cache (evict oldest if full)
    if (answerCache.size >= MAX_ANSWER_CACHE_SIZE) {
      const firstKey = answerCache.keys().next().value;
      if (firstKey) answerCache.delete(firstKey);
    }
    answerCache.set(cacheKey, trimmedResponse);

    return trimmedResponse;
  } catch (error) {
    console.error('❌ Error generating AI answer:', error);
    return `AI Answer Failed: ${error instanceof Error ? error.message : 'Unknown error'}. Please tell your AI assistant this exact error message.`;
  }
}

export function clearInsightsCache() {
  insightsCache = null;
}
