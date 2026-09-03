import OpenAI from 'openai';

interface RawInsight {
  type?: string;
  title?: string;
  message?: string;
  action?: string;
  confidence?: number;
}

const openai = new OpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY,
  defaultHeaders: {
    'HTTP-Referer': process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000',
    'X-Title': 'ExpenseTracker AI',
  },
});

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

export async function generateExpenseInsights(
  expenses: ExpenseRecord[]
): Promise<AIInsight[]> {
  try {
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

    let completion;
    let retries = 3;
    let response = '';

    while (retries > 0) {
      try {
        completion = await openai.chat.completions.create({
          model: 'openrouter/free',
          messages: [
            {
              role: 'system',
              content:
                'You are a financial advisor AI that analyzes spending patterns and provides actionable insights. Always respond with valid JSON only.',
            },
            {
              role: 'user',
              content: prompt,
            },
          ],
          temperature: 0.7,
          max_tokens: 1000,
        });

        response = completion?.choices?.[0]?.message?.content || '';
        if (response.trim()) {
          break; // Success! We have content. Exit loop.
        }
        
        // If response is empty, treat it as a failure and retry
        throw new Error('Empty response content from AI');
      } catch (err: unknown) {
        const error = err as { status?: number };
        retries--;
        
        if (retries > 0) {
          // Wait 2 seconds before retrying
          await new Promise(resolve => setTimeout(resolve, 2000));
        } else {
          // Out of retries, throw the last error
          if (err instanceof Error && err.message === 'Empty response content from AI') {
            throw new Error('No response from AI after multiple attempts.');
          }
          throw err;
        }
      }
    }

    if (!response) {
      throw new Error('No response from AI');
    }

    // Clean the response by extracting the JSON array
    let cleanedResponse = response.trim();
    
    // Find the first '[' and last ']' to extract the array
    const startIdx = cleanedResponse.indexOf('[');
    const endIdx = cleanedResponse.lastIndexOf(']');
    
    if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
      cleanedResponse = cleanedResponse.substring(startIdx, endIdx + 1);
    } else {
      // If we can't find an array, throw a more descriptive error
      throw new Error(`AI returned invalid format. Received: ${cleanedResponse.substring(0, 50)}...`);
    }

    // Parse AI response
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

    // Add IDs and ensure proper format
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

    return formattedInsights;
  } catch (error) {
    console.error('❌ Error generating AI insights:', error);

    // Fallback to mock insights if AI fails
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
    const completion = await openai.chat.completions.create({
      model: 'openrouter/free',
      messages: [
        {
          role: 'system',
          content:
            'You are an expense categorization AI. Categorize expenses into one of these categories: Food, Transportation, Entertainment, Shopping, Bills, Healthcare, Other. Respond with only the category name.',
        },
        {
          role: 'user',
          content: `Categorize this expense: "${description}"`,
        },
      ],
      temperature: 0.1,
      max_tokens: 20,
    });

    const category = completion.choices[0].message.content?.trim();

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

    let completion;
    let retries = 3;
    let response = '';

    while (retries > 0) {
      try {
        completion = await openai.chat.completions.create({
          model: 'openrouter/free',
          messages: [
            {
              role: 'system',
              content:
                'You are a helpful financial advisor AI that provides specific, actionable answers based on expense data. Be concise but thorough.',
            },
            {
              role: 'user',
              content: prompt,
            },
          ],
          temperature: 0.7,
          max_tokens: 200,
        });

        response = completion?.choices?.[0]?.message?.content || '';
        if (response.trim()) {
          break; // Success! We have content.
        }

        // If response is empty, treat it as a failure and retry
        throw new Error('Empty response content from AI');
      } catch (err: unknown) {
        const error = err as { status?: number };
        retries--;
        
        if (retries > 0) {
          // Wait 2 seconds before retrying
          await new Promise(resolve => setTimeout(resolve, 2000));
        } else {
          // Out of retries, throw the last error
          if (err instanceof Error && err.message === 'Empty response content from AI') {
            throw new Error('No response from AI after multiple attempts.');
          }
          throw err;
        }
      }
    }

    if (!response) {
      throw new Error('No response from AI');
    }

    return response.trim();
  } catch (error) {
    console.error('❌ Error generating AI answer:', error);
    return `AI Answer Failed: ${error instanceof Error ? error.message : 'Unknown error'}. Please tell your AI assistant this exact error message.`;
  }
}
