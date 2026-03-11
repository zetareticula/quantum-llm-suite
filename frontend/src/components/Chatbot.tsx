import React, { useState } from 'react';
import axios from 'axios';
import { InferenceResponse } from '../types';

const Chatbot: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState<InferenceResponse | null>(null);
  const [apiKey, setApiKey] = useState('');
  const [quantumProvider, setQuantumProvider] = useState('IBM');
  const [cloudOffering, setCloudOffering] = useState('AWS');
  const [model, setModel] = useState('Qwen/Qwen2.5-72B-Instruct');
  const [quantLevel, setQuantLevel] = useState('fp16');
  const [quantMode, setQuantMode] = useState('classical');
  const [useReal, setUseReal] = useState(false);

  const handleSubmit = async () => {
    try {
      const res = await axios.post('http://localhost:8000/inference', {
        prompt,
        quantum_provider: quantumProvider,
        cloud_offering: cloudOffering,
        model,
        quantization_level: quantLevel,
        quantization_mode: quantMode,
        use_quantum: true,
        use_real_hardware: useReal
      }, { headers: { 'api-key': apiKey } });
      setResponse(res.data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <input type="text" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="API Key" />
      <select value={quantumProvider} onChange={e => setQuantumProvider(e.target.value)}>
        <option>IBM</option>
        <option>Rigetti</option>
        <option>Google Quantum AI</option>
        <option>IonQ</option>
        <option>Local</option>
        <option>AWS Braket</option>
      </select>
      <select value={cloudOffering} onChange={e => setCloudOffering(e.target.value)}>
        <option>AWS</option>
        <option>Google Cloud</option>
        <option>Azure</option>
        <option>Oracle Cloud</option>
        <option>IBM Cloud</option>
      </select>
      <select value={model} onChange={e => setModel(e.target.value)}>
        <option>Qwen/Qwen2.5-72B-Instruct</option>
        <option>meta-llama/Meta-Llama-3.1-70B-Instruct</option>
        <option>mistralai/Mistral-Large-Instruct-2407</option>
        <option>THUDM/glm-4-9b-chat</option>
        <option>DeepSeek-AI/DeepSeek-V2-Chat</option>
      </select>
      <select value={quantLevel} onChange={e => setQuantLevel(e.target.value)}>
        <option>fp16</option>
        <option>int8</option>
        <option>int4</option>
      </select>
      <select value={quantMode} onChange={e => setQuantMode(e.target.value)}>
        <option>classical</option>
        <option>quantum_distillation</option>
        <option>quantum_embedding</option>
      </select>
      <input type="checkbox" checked={useReal} onChange={e => setUseReal(e.target.checked)} /> Use Real Hardware (if token set)
      <input type="text" value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="Enter prompt" />
      <button onClick={handleSubmit}>Submit</button>
      {response && (
        <div>
          <p>{response.classical_output} (Time: {response.time_classical}s)</p>
          <p>{response.quantum_output} (Time: {response.time_quantum}s)</p>
          <p>Benchmark: {response.benchmark}</p>
        </div>
      )}
    </div>
  );
};

export default Chatbot;