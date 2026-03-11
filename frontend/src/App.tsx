import React from 'react';
import Chatbot from './components/Chatbot';

const App: React.FC = () => {
  return (
    <div className="App">
      <header>Quantum LLM Chatbot</header>
      <Chatbot />
    </div>
  );
};

export default App;