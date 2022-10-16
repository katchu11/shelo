import { useCallback } from "react";

import "survey-core/defaultV2.min.css";
import { StylesManager, Model } from "survey-core";
import { Survey } from "survey-react-ui";
import axios from 'axios'

import {useEffect, useState} from 'react';


StylesManager.applyTheme("defaultV2");

const surveyJson = {
  elements: [
    {"type":"panel",
  "name":"panel0","elements":[{
    "type": "rating",
    "name": "question2",
    "title": "How many cups of water did you drink?",
    "rateMax": 12,
    "minRateDescription": "A little",
    "maxRateDescription": "A lot"
  },
  {
    "type": "rating",
    "name": "question3",
    "title": "How many hours of sleep did you get last night?",
    "rateMax": 10,
    "minRateDescription": "A little",
    "maxRateDescription": "A lot"
  },
  {
    "type": "boolean",
    "name": "question4",
    "title": "Did you drink any alcohol today?"
  }, {
    "type": "boolean",
    "name": "question5",
    "title": "Did you take your necessary medications today?"
     },]},
     {
      "type": "panel",
      "name": "panel1",
      "elements": [
        {
          "type": "rating",
          "name": "question6",
          "title": "How long were you physicaly active for today?",
          "rateMax": 10,
          "minRateDescription": "A little",
          "maxRateDescription": "A lot"
        },
        {
          "type": "rating",
          "name": "question7",
          "title": "How good was your sleep last night?",
          "rateMax": 5,
          "minRateDescription": "Great",
          "maxRateDescription": "Terrible"
        },
        {
          "type": "rating",
          "name": "question8",
          "title": "How would you rate your stress levels?",
          "rateMax": 5,
          "minRateDescription": "High",
          "maxRateDescription": "Low"
        },
        {
          "type": "boolean",
          "name": "question9",
          "title": "Did you experience any numbness in your hands or feet today?"
        }, {
          "type": "rating",
          "name": "question10",
          "title": "How many times did you have to urinate in the night before?",
                    "rateMax": 5,
          "minRateDescription": "A little",
          "maxRateDescription": "A lot"
        }, {
          "type": "boolean",
          "name": "question11",
          "title": "Did you experience any low blood-sugar reactions today?",
          }, {
          "type": "rating",
          "name": "question12",
          "title": "How would you rate your overall mood today?",
          "rateMax": 5,
          "minRateDescription": "Great",
          "maxRateDescription": "Terrible"
        }
      ],
      "visible": true,
    }, 
    {
      "type": "file",
      "title": "Please upload your photo",
      "name": "image",
      "storeDataAsText": false,
      "showPreview": true,
      "imageWidth": 150,
      "maxSize": 102400000
    }
  ]
};


// You can store file id as a content property of a file question value
// In this case you should handle both `onUploadFiles` and `onDownloadFile` events
export function Test() {
  
}
function App() {
  const survey = new Model(surveyJson);
  survey.focusFirstQuestionAutomatic = false;
//   const response = axios.get('localhost:5000/', {
// })
const [data, setData] = useState();
  // useEffect(() => {
  //   console.debug("running once");
  //   fetch("http://127.0.0.1:5000").then(response => console.debug("response", response))
  // }, [])
    const alertResults = useCallback((sender) => {
    // const results = JSON.stringify(sender.data);
    fetch("http://127.0.0.1:5000").then(response => console.log(response));
    // console.log(response)
  }, []);

  survey.onComplete.add(alertResults);
  survey.onUploadFiles.add(function (survey, options) {
    var formData = new FormData();
    options.files.forEach(function (file) {
      formData.append(file.name, file);
    });

    options.callback("success", options.files.map(function (file) {
        return {
          file: file.name,
          content: "hi"
        };
      }));
    ;
  });
  
  return <Survey model={survey} />;
  
}

export default App;
