export interface InferenceResponse {
  classical_output: string;
  quantum_output: string;
  benchmark: string;
  time_classical: number;
  time_quantum: number;
}