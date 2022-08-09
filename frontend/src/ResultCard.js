import React from 'react'
import { useState, useEffect } from "react";
import { Row, Col, Card } from "react-bootstrap";
import "./App.css";

const ResultCard = ({ result }) => {
	const [expanded, setExpanded] = useState(false);
	const [show, setShow] = useState(false);

	const words = result.plot.split(" ")

	const shortenText = (text) => {
		return words.slice(0,50).join(" ")
	}

	useEffect(() => {
		if (words.length > 50) {
			setShow(true)
		} 
	}, [words.length])

  return (
    <>
			<Card>
				<Card.Body>
					<Card.Title>{result.title}</Card.Title>
					<Card.Subtitle>Score: {result.score}</Card.Subtitle>
					<Card.Text>{expanded ? result.plot : shortenText(result.plot)}</Card.Text>
					{show && <span className="read-more" onClick={() => setExpanded(!expanded)}>{expanded ? "Read Less" : "Read More"}</span>}
				</Card.Body>
			</Card>
    </>
  )
}

export default ResultCard