import logo from "./logo.svg";
import "./App.css";
import React, { useState } from "react";
import HeaderWrapper from "./components/Header/HeaderWrapper";
import NavBar from "./components/Header/NavBar";
import Logo from "./components/Header/Logo";
import FeatureWrapper from "./components/Header/FeatureWrapper";
import FeatureTitle from "./components/Header/FeatureTitle";
import Warning from "./components/Header/Warning";
import { Row, Col, Button, InputGroup, Form, Card } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import axios from "axios";

function App() {
  const [searchQuery, setSearchQuery] = useState("");
  const [engine, setEngine] = useState("");
  const [searchResults, setSearchResults] = useState([]);

  const onChangeQuery = (e) => {
    setSearchQuery(e.target.value);
  };

  const onSubmit = (e) => {
    e.preventDefault();
    console.log(searchQuery);
    axios
      .get("http://localhost:5000/search", {
        params: { engine: "BM25", query: searchQuery },
      })
      .then((res) => {
        console.log("SUCCESS", res);
        setSearchResults(res.data);
        console.log(searchQuery);
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
          <NavBar className="navbar-home">
            <Logo />
          </NavBar>
          <FeatureWrapper className="feature-wrapper-home">
            <FeatureTitle className="feature-title-home">
              Unlimited movies, TV shows and more.
            </FeatureTitle>
            <Warning>Search for your favourite movies!</Warning>

            <InputGroup style={{ paddingLeft: "3rem", paddingRight: "3rem" }}>
              <Form.Control
                placeholder="Search for your favourite movies!"
                onChange={onChangeQuery}
              />
              <Button variant="danger" onClick={onSubmit}>
                Search
              </Button>
            </InputGroup>
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
              <Card>
                <Card.Body>
                  <Card.Title>Movie Title</Card.Title>
                  <Card.Subtitle>Score: 100</Card.Subtitle>
                  <Card.Text>This is a random movie plot.</Card.Text>
                </Card.Body>
              </Card>
            </>
          ))}
        </Col>
      </Row>
    </HeaderWrapper>
  );
}

export default App;
