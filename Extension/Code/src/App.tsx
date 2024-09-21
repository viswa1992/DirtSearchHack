import { useState, useEffect} from 'react'
import factCheckLogo from './assets/cloud-network.gif'
import './App.css'
import HashLoader from "react-spinners/ClockLoader";
import redflagLogo from './assets/red-flag.gif'
//import puzzleLogo from './assets/puzzle.gif'
import safeLogo from './assets/safe.gif'
import axios from 'axios';

const styles = {
  container: {
    margin: 'auto',
    display: 'inline-flex',
    alignItems: 'center', // Aligns items vertically in the center
  },
  heading: {
    fontFamily: 'Arial, sans-serif',
    color: 'black',
    display: 'flex',
    alignItems: 'center', // Aligns text and image vertically in the center
  },
  logo: {
    marginLeft: '8px', // Optional: Adds some space between text and logo
  },
};

async function getSyncStorage(keys: any) {
  return new Promise((resolve, reject) => {
      chrome.storage.sync.get(keys, (result) => {
          if (chrome.runtime.lastError) {
              reject("Error getting data from chrome storage");
          } else {
              resolve(result);
          }
      });
  });
}

function App() {

  const [claim, setClaim] = useState('');
  const [loader, setLoader] = useState(true);
  const [fact, setFact] = useState<any>();

  async function getFact(data: any) {
    console.log(data);
    const response: any = await axios.post('https://dirtsearchgpt.azurewebsites.net/process-text',{text: data});
    // const response: any = await axios.post('http://127.0.0.1:8000/process-text',{text: data});
    console.log(response.data);
    setFact(response.data);
    console.log(fact);
  }

  async function initialize() {
    const storage:any = await getSyncStorage('text');
    setClaim(storage.text);
    await getFact(storage.text);
    setLoader(false);
  }

  useEffect(() => {
    initialize();
  }, []);


  if(loader) {
    
    return (
    <>
      {claim && <>
      <div style={{padding: '2rem'}}>
        <div style={styles.container}>
          <h2 style={styles.heading}>
            Reality Checker
            <img src={factCheckLogo} width="50" height="50" alt="Fact Check Logo" style={styles.logo} />
          </h2>
        </div>
        <div>
        <p className="read-the-docs">
          <b>Claim:</b> {claim}
        </p>
        </div>
      </div>
      <div className="card" style={{paddingTop: 0}}>
      <HashLoader
          color = {"#1fec94"}
          loading={loader}
          size={70}
          cssOverride={{margin: 'auto'}}
          aria-label="Loading Spinner"
          data-testid="loader"
        />
    </div>
      </>}
    </>
      
    )
  }

  return (
    <>
    <div style={{width: '400px'}}>
      <div style={styles.container}>
        <h2 style={styles.heading}>
          {fact ?
          <>
           {(fact && fact.status== "Low") &&  <> <img src={redflagLogo} width="50" height="50" alt="Fact Check Logo" style={styles.logo} /> Claim rating: Low </>}
           {(fact && fact.source_used == "None") && <> <img src={safeLogo} width="50" height="50" alt="Fact Check Logo" style={styles.logo} /> No claim to validate</>}
           {(fact && fact.status == "High") &&  <> <img src={safeLogo} width="50" height="50" alt="Fact Check Logo" style={styles.logo} /> Claim rating: High</>}
           {(fact && fact.status == "Medium") &&  <> <img src={redflagLogo} width="50" height="50" alt="Fact Check Logo" style={styles.logo} /> Claim rating: Medium</>}
          </> 
           : 
           "Facing issues please try again after some time"}

        </h2>
      </div>
      <div className="card" style={{paddingTop: 0 }}>
        <p className="read-the-docs">
          <b>Source used: </b> {fact.source_used}
        </p>
        <p className="read-the-docs">
          <i>{fact.result}</i>
        </p>

        {fact && fact.search_results && fact.search_results.length > 0 &&<>
        {        
        <p className="read-the-docs">
          <b>Check the following urls for more information:</b>
        </p>}
        {fact.search_results.map((item: any) => {
          console.log(item.query);  
          if(item  && item.results.length > 0) {
            return (
              <p className="read-the-docs">
                <a href={item.results[0].url} target='_blank'>{item.results[0].title}</a>
                </p>)
          }
        })} </>}
      </div>
    </div>
    </>
  )
}

export default App
