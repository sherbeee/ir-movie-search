import logo from "./logo.svg";
import "./App.css";
import React from "react";
import HeaderWrapper from "./components/Header/HeaderWrapper";
import NavBar from "./components/Header/NavBar";
import Logo from "./components/Header/Logo";
import FeatureWrapper from "./components/Header/FeatureWrapper";
import FeatureTitle from "./components/Header/FeatureTitle";
import Warning from "./components/Header/Warning";
import { Row, Col, Button, InputGroup, Form, Card } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
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

            <InputGroup style={{ paddingLeft: '3rem', paddingRight: '3rem'}} >
              <Form.Control
                placeholder="Search for your favourite movies!"
              />
              <Button variant="danger">
                Button
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
          <Card>
            <Card.Body>
              <Card.Title>Movie Title</Card.Title>
              <Card.Subtitle>Score: 100</Card.Subtitle>
              <Card.Text>This is a random movie plot.</Card.Text>
            </Card.Body>
            </Card>
        </Col>
      </Row>
    </HeaderWrapper>
  );
}

export default App;
