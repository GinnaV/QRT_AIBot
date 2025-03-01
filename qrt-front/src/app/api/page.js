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

  // calls getAPI
  const fetchData = async () => {
    try {
      const data = await getApi();
      document.getElementById('output').textContent = data;
      return data;
    } catch (error) {
      setError(error.message);
    }
  };

const Page = () => {
    const router = useRouter();
    const [formData, setFormData] = useState({ name: "" });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
  
    // form submission
    const onFinish = (event) => {
      event.preventDefault();
      setIsLoading(true);
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
      <form onSubmit={onFinish} >
        <div className={styles.main}>
        <div className={styles.ctas}>
          <label htmlFor="query">Enter your query here</label>
          <input
            required
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
        <div id="output"></div>
      </form>
    );
  };
  
  export default Page;