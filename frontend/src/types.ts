export interface Source {
  rel_path: string;
  heading_path: string[];
  snippet: string;
  score: number;
}

export interface ChatMessage {
  id: string;
  question: string;
  answer: string;
  sources: Source[];
  streaming: boolean;
  error?: string;
}
