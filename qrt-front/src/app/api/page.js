"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import styles from "../page.module.css";

/* To run application:
cd qrt-front && npm run dev
Navigate to http://localhost:3000/api
*/

// gets data from API
async function getApi() {
    const res = await fetch(`https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=bitcoin&x_cg_demo_api_key=CG-WV56B4Cuz7isjJncGV79YQZW`);
    if (!res.ok) {
      throw new Error("Failed to retrieve api details");
    }
    return res.text(); // could also return res.json() here if we want it in the json format
  }

// posts data to API and returns response
  async function postApi(data) {
    const res = await fetch(`https://httpbin.org/post`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: data,
    });
  
    if (!res.ok) {
      throw new Error("Failed to update menu");
    }
    return res.text();
  }

  // calls getAPI
  const fetchData = async () => {
    try {
      const data = await getApi();
      //const data = await postApi(document.getElementById('query').value);
      messageDisplay(data, document.getElementById('answer'), 15);
      return data;
    } catch (error) {
      setError(error.message);
    }
  };

  function messageDisplay(str, element, timeBetween) {
    var index = -1;
    (function go() {
      if (++index < str.length) {
        element.innerHTML = element.innerHTML + str.charAt(index);
        setTimeout(go, timeBetween);
      }
    })();
    }

const Page = () => {
    const router = useRouter();
    const [formData, setFormData] = useState({ name: "" });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
  
    // form submission
    const onFinish = (event) => {
      event.preventDefault();
      setIsLoading(true);

      document.getElementById("question").style.display = "block";
      document.getElementById("answer").style.display = "block";
      document.getElementById('question').textContent = document.getElementById('query').value;
      document.getElementById('query').value = "";
      fetchData()
        .then(result => {
          // can perform operations with the result of fetchData here
          console.log(result);
        })
        .catch(() => {
          setError("An error occurred");
          setIsLoading(false);
        });
    };
  
    // cleanup effect for resetting loading state
    useEffect(() => {
      return () => setIsLoading(false);
    }, []);

    return (
      <form id="form" onSubmit={onFinish} >
        <div className={styles.main}>
        <div className={styles.ctas}>
          <label htmlFor="query">Enter your query here:</label>
          <input
            required
            id="query"
            name="query"
            value={formData.query}
            onChange={(event) =>
              setFormData({ ...formData, name: event.target.value })
            }
          />
        </div>
        {error && <p className="error-message">{error}</p>}
        <div>
          <button disabled={isLoading} className={styles.primary} type="submit">
            Submit
          </button>
        </div>
        </div>
        <div className={styles.question} id="question" style={{display: "none"}}></div>
        <div className={styles.answer} id="answer" style={{display: "none"}}></div>
      </form>
    );
  };
  
  export default Page;