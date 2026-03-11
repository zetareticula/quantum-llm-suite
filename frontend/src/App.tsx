import React from 'react';
import Chatbot from './components/Chatbot';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col">
      <header className="px-6 py-4 border-b border-gray-700">
        <h1 className="text-xl font-semibold tracking-tight">Quantum LLM Suite</h1>
      </header>
      <main className="flex-1 overflow-hidden">
        <Chatbot />
      </main>
    </div>
  );
};

export default App;
