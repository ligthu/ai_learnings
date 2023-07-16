import Head from "next/head";
import { useState } from "react";
import styles from "./index.module.css";

export default function Home() {
  const [query_input, set_query_input] = useState("");
  const [model_name, set_model_name] = useState("text-davinci-003");
  const [results, set_results] = useState([]);

  async function onSubmit(event) {
    event.preventDefault();
    try {
      const model_name_fixed = model_name;
      const response = await fetch("/api/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query_input, model: model_name_fixed }),
      });

      const data = await response.json();
      if (response.status !== 200) {
        throw (
          data.error ||
          new Error(`Request failed with status ${response.status}`)
        );
      }

      const res = `(${model_name_fixed}): ${query_input}\n${JSON.stringify(
        data.result
      )}\n`;
      if (results.length == 0 || results[results.length - 1] != res) {
        set_results([...results, res]);
      }
      set_query_input("");
    } catch (error) {
      console.error(error);
      alert(error.message);
    }
  }

  const codeStyle = {
    whiteSpace: "pre",
    backgroundColor: "#eee",
    fontFamily: "Consolas, monospace",
    fontSize: "14px",
    lineHeight: "1.4",
    color: "#333",
    display: "block",
    padding: "20px",
  };

  return (
    <div>
      <Head>
        <title>Auto Insights</title>
      </Head>

      <main className={styles.main}>
        <h3>Query airports data</h3>
        <div>
          <label htmlFor="model_name"> GPT model being used </label>
          <select
            id="model_name"
            value={model_name}
            onChange={(e) => set_model_name(e.target.value)}
          >
            <option value="text-davinci-003">text-davinci-003</option>
            <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
            <option value="gpt-4">gpt-4</option>
          </select>
        </div>
        <form onSubmit={onSubmit}>
          <input
            type="text"
            name="query_text"
            placeholder="What do you want to know about airports?"
            value={query_input}
            onChange={(e) => set_query_input(e.target.value)}
          />
          <input type="submit" value="Generate insights" />
        </form>
        <code style={codeStyle}>{results}</code>
      </main>
    </div>
  );
}
