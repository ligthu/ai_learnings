const sqlite3 = require("sqlite3").verbose();

const { Configuration, OpenAIApi } = require("openai");
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

// true => chat, flase => completion
const engines = {
  "text-davinci-003": false,
  "gpt-3.5-turbo": true,
  "gpt-4": true,
};

function extract_code_from_text(text) {
  const code_regex = /```( ?sql)?([\s\S]*?)```/g;
  const matches = text.matchAll(code_regex);
  const code_blocks = [];

  for (const match of matches) {
    const code_block = match[2];
    code_blocks.push(code_block.trim("\n").trim());
  }

  return code_blocks;
}

function expertise() {
  return `
	You are an expert data scientist who knows sql very well.
	You can be provided table schemas and details questions 
	to get insights from data and you can create accurate sql queries 
	to extract these insights.
	`;
}

function construct_query(query) {
  return `
    "airports" database has the following tables
    tbl_2004_final_statistics
    tbl_2007_final_statistics
    tbl_2008_final_statistics
    tbl_2009_final_statistics
    tbl_2010_final_statistics
    tbl_2011_preliminary_statistics
    tbl_2012_preliminary_statistics

    tbl_2004_final_statistics has the following schema
          Rank INTEGER,
          Airport TEXT,
          Code TEXT, -- this is in upper case
          Location TEXT, -- this has country name
          Total_Cargo REAL,
          Prev_Rank INTEGER -- rank for the previous year, 2003
          Percentage_Change REAL

    all the other tables have the following schema
          Rank INTEGER,
          Airport TEXT,
          Code TEXT, -- this is in upper case
          Total_Cargo REAL,
          Percentage_Change REAL

    ---
    do not explain only provide sql for:
    ${query}
    `;
}

// Use OpenAI api api to convert NLP to SQL
async function make_sql(query, model) {
  let answer = "";
  if (engines[model]) {
    const response = await openai.createChatCompletion({
      model: model,
      messages: [
        {
          role: "system",
          content: expertise(),
        },
        {
          role: "user",
          content: construct_query(query),
        },
      ],
      temperature: 0,
      max_tokens: 1024,
      top_p: 1.0,
      stop: ["You:"],
    });
    answer = response.data.choices[0].message.content;
  } else {
    const response = await openai.createCompletion({
      model: model,
      prompt: construct_query(query),
      temperature: 0,
      max_tokens: 1024,
      top_p: 1.0,
      stop: ["You:"],
    });
    answer = response.data.choices[0].text;
  }

  const queries = extract_code_from_text(answer);
  if (queries.length == 0) {
    return [answer];
  }
  return queries;
}

// Function to execute a query on sqlite db
function execute_query(db, query, params) {
  return new Promise((resolve, reject) => {
    db.all(query, params, (err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(rows);
      }
    });
  });
}

// Fetch data from sqlite db
async function fetch_data(query, model) {
  const db = new sqlite3.Database("./pages/api/db/airports.db");
  let sql_blocks = await make_sql(query, model);
  let results = [];
  for (const sql_block of sql_blocks) {
    console.log(`${model}:\n${sql_block}`);
    const result = await execute_query(db, sql_block);
    results.push(result);
  }

  db.close();

  return results;
}

export default async function (req, res) {
  if (!configuration.apiKey) {
    res.status(500).json({
      error: {
        message:
          "OpenAI API key not configured, please follow instructions in README.md",
      },
    });
    return;
  }

  let query_text = req.body.query.trim();
  let model_name = req.body.model.trim();
  if (query_text.length === 0) {
    res.status(400).json({
      error: {
        message: "Please enter a valid query",
      },
    });
    return;
  }

  try {
    const result = await fetch_data(query_text, model_name);
    console.log(result);
    res.status(200).json({ result: result });
  } catch (error) {
    // Consider adjusting the error handling logic for your use case
    if (error.response) {
      console.error(error.response.status, error.response.data);
      res.status(error.response.status).json(error.response.data);
    } else {
      console.error(`Error with OpenAI API request: ${error.message}`);
      res.status(500).json({
        error: {
          message: "An error occurred during your request.",
        },
      });
    }
  }
}
