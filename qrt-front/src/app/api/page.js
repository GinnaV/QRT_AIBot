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

var messageId = 0;

  function sendDataCallback() {
    // Check response is ready or not
    if (xhr.readyState == 4 && xhr.status == 201) {
      const newDiv = document.createElement('div');
      newDiv.innerHTML += `<div class="answer" id="answer${messageId}" ></div>`;
      document.getElementById("mainDiv").appendChild(newDiv);
        messageDisplay(JSON.parse(xhr.responseText).message, document.getElementById('answer'+messageId), 10);
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

function submit() {
  //document.getElementById("question").style.display = "block";
  var question = document.getElementById('query').value;
      //document.getElementById("answer").style.display = "block";
      document.getElementById("answer").innerHTML = "";
      messageId += 1;
      const newDiv = document.createElement('div');
      newDiv.innerHTML += `<div class="question" id="question${messageId}" >${question}</div>`;
      document.getElementById("mainDiv").appendChild(newDiv);
      //document.getElementById('question').textContent = question;
      sendData(question);
}

const Page = () => {
  
    // form submission
    const onFinish = (event) => {
      document.getElementById("question").style.display = "block";
      document.getElementById("answer").style.display = "block";
      document.getElementById("answer").innerHTML = "";
      var question = document.getElementById('query').value;
      document.getElementById('question').textContent = question;
      messageId += 1;
      document.getElementById("mainDiv")
      .innerHTML += `<div className=${styles.question} id="question${messageId}" >${question}</div>`;
      sendData(question);
    };

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
          <button className={styles.primary} id="submit" onClick={submit}>
            Submit
          </button>
        </div>
        <div className={styles.question} id="question" style={{display: "none"}}></div>
        <div className={styles.answer} id="answer" style={{display: "none"}}></div>
        </div>

    );
  };
  
  export default Page;