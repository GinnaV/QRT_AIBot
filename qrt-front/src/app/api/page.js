"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import styles from "../page.module.css";

/* To run application:
cd qrt-front && npm run dev
Navigate to http://localhost:3000/api
*/
var xhr = null;

    const getXmlHttpRequestObject = function () {
        if (!xhr) {
            // Create a new XMLHttpRequest object 
            xhr = new XMLHttpRequest();
        }
        return xhr;
    };

  function sendDataCallback() {
    // Check response is ready or not
    if (xhr.readyState == 4 && xhr.status == 201) {
        messageDisplay(JSON.parse(xhr.responseText).message, document.getElementById('answer'), 10);
    }
}

  function sendData(question) {
    document.getElementById('query').value = "";

    xhr = getXmlHttpRequestObject();
    xhr.onreadystatechange = sendDataCallback;
    // asynchronous requests
    xhr.open("POST", "http://localhost:6969/users", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.setRequestHeader('Access-Control-Allow-Headers', '*');
    xhr.setRequestHeader('Access-Control-Allow-Origin', '*');
    // Send the request over the network
    xhr.send(JSON.stringify({"data": question}));
}

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
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
  
    // form submission
    const onFinish = (event) => {

      document.getElementById("question").style.display = "block";
      document.getElementById("answer").style.display = "block";
      document.getElementById("answer").innerHTML = "";
      var question = document.getElementById('query').value;
      document.getElementById('question').textContent = question;
      sendData(question);
    };
  
    // cleanup effect for resetting loading state
    useEffect(() => {
      return () => setIsLoading(false);
    }, []);

    return (
        <div id="mainDiv" className={styles.main}>
        <a href="tel:+07700160292">Call the AI bot instead: 07700160292</a>
        <div className={styles.ctas}>
          <label htmlFor="query">Enter your query here:</label>
          <input
            required
            id="query"
            name="query"
          />
        </div>
        {error && <p className="error-message">{error}</p>}
        <div>
          <button disabled={isLoading} className={styles.primary} type="submit" onClick={onFinish}>
            Submit
          </button>
        </div>
        <div className={styles.question} id="question" style={{display: "none"}}></div>
        <div className={styles.answer} id="answer" style={{display: "none"}}></div>
        </div>

    );
  };
  
  export default Page;