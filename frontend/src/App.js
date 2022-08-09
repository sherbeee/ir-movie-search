import logo from "./logo.svg";
import "./App.css";
import React, { useEffect, useState } from "react";
import HeaderWrapper from "./components/Header/HeaderWrapper";
import NavBar from "./components/Header/NavBar";
import Logo from "./components/Header/Logo";
import FeatureWrapper from "./components/Header/FeatureWrapper";
import FeatureTitle from "./components/Header/FeatureTitle";
import Warning from "./components/Header/Warning";
import { Row, Col, Button, InputGroup, Form } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import axios from "axios";
import ResultCard from "./ResultCard";

function App() {
  const [searchQuery, setSearchQuery] = useState("");
  const [engine, setEngine] = useState("BM25");
  const [searchResults, setSearchResults] = useState([]);
  const [showLoader, setShowLoader] = useState(false);
  const [showWarning, setWarning] = useState(false);
  
  const onChangeQuery = (e) => {
    setSearchQuery(e.target.value);

    if (e.target.value.length > 0) {
      setWarning(false);
    }
  };

  const onEngineChange = (e) => {
    setEngine(e.target.value);
  }

  const onSubmit = (e) => {
    e.preventDefault();
    
    if (searchQuery.length === 0) {
      setWarning(true)
      return
    }

    setWarning(false)
    setSearchResults([]);
    setShowLoader(true);
    console.log({engine: engine, query: searchQuery});
    axios
      .get("http://127.0.0.1:5000/search", {
        params: { engine: engine, query: searchQuery },
      })
      .then((res) => {
        console.log("SUCCESS", res.data);
        setShowLoader(false);
        setSearchResults(res.data.results);
        /*axios
      .get("http://localhost:5000/test", {params: {engine: "BM25", query: searchQuery}})
      .then((res) => {
        console.log("SUCCESS", res);*/
      })
      .catch((err) => {
        console.log("ERROR", err);
      });
  };

  return (
    <HeaderWrapper className="header-wrapper-home">
      <Row>
        <Col md="6">
          <NavBar className="navbar-home"></NavBar>
          <FeatureWrapper className="feature-wrapper-home">
            <FeatureTitle className="feature-title-home">
              Unlimited movies, TV shows and more.
            </FeatureTitle>
            <Warning>Search for your favourite movies!</Warning>
            <Col style={{ paddingLeft: "3rem", paddingRight: "3rem" }}>
              <InputGroup  style={{marginBottom:"20px"}}>
                <InputGroup.Text>Search Engine:
                </InputGroup.Text>
                <Form.Select onChange={onEngineChange}>
                  <option>BM25</option>
                  <option>BM25 with BERT rerank</option>
                  <option>S-BERT</option>
                  <option>S-BERT with Cross-Encoder Reranking</option>
                </Form.Select>
              </InputGroup>


              <InputGroup >
                <Form.Control
                  placeholder="Search for your favourite movies!"
                  onChange={onChangeQuery}
                />
                <Button variant="danger" onClick={onSubmit}>
                  Search
                </Button>
              </InputGroup>
              {showWarning && <h6 className="query-warning">Please input a query!</h6>}
            </Col>

            <br />
            <br />
            <br />
            <br />
            <br />
            <br />
            <br />
            <br />
            <br />
          </FeatureWrapper>
        </Col>
        <Col md="6">
          {searchResults.map((result) => (
            <>
              <ResultCard result={result}></ResultCard>
            </>
          ))}
          {showLoader && <h6 className="loading">Loading Results...</h6>}
        </Col>
      </Row>
    </HeaderWrapper>
  );
}

export default App;
