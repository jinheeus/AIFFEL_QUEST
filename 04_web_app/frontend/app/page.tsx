'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Send, ChevronDown, ChevronRight, Loader2, StopCircle } from 'lucide-react';

// --- Types ---
type Message = {
    role: 'user' | 'assistant';
    content: string;
    thoughtProcess?: ThoughtStep[];
};

type ThoughtStep = {
    node: string;
    content: string;
    status: 'pending' | 'done';
};

export default function Home() {
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [currentThoughts, setCurrentThoughts] = useState<ThoughtStep[]>([]);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // [New] Unique Session ID per page refresh
    // Using useRef with lazy initialization to ensure it's created once per mount
    const sessionId = useRef<string>('');
    if (!sessionId.current) {
        sessionId.current = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    }

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(scrollToBottom, [messages, currentThoughts]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim() || isLoading) return;

        // [New] Generate Session ID if not present (although useState initializer handles it)
        // Ensure we use the current sessionId state
        const currentSessionId = sessionId.current;

        const history = messages.map(m => ({ role: m.role, content: m.content }));
        const userMsg: Message = { role: 'user', content: query };

        setMessages(prev => [...prev, userMsg]);
        setQuery('');
        setIsLoading(true);
        setCurrentThoughts([]);

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: userMsg.content,
                    // persona field removed
                    history: history,
                    session_id: currentSessionId, // [New] Pass unique session ID
                }),
            });

            if (!response.body) throw new Error("No response body");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantMsg: Message = { role: 'assistant', content: '', thoughtProcess: [] };

            setMessages(prev => [...prev, assistantMsg]);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.replace('data: ', '').trim();
                        if (dataStr === '[DONE]') {
                            setIsLoading(false);
                            break;
                        }

                        try {
                            const data = JSON.parse(dataStr);
                            if (data.type === 'status') {
                                setCurrentThoughts(prev => {
                                    const exists = prev.find(t => t.node === data.node);
                                    if (!exists && data.node) {
                                        return [...prev, { node: data.node, content: data.content, status: 'done' }];
                                    }
                                    return prev;
                                });
                            } else if (data.type === 'answer') {
                                assistantMsg.content = data.content;
                                setMessages(prev => {
                                    const newMsgs = [...prev];
                                    newMsgs[newMsgs.length - 1] = { ...assistantMsg, thoughtProcess: [...currentThoughts] }; // Bake in latest thoughts
                                    return newMsgs;
                                });
                            }
                        } catch (err) { }
                    }
                }
            }
            // Finalize thoughts
            setMessages(prev => {
                const newMsgs = [...prev];
                const lastMsg = newMsgs[newMsgs.length - 1];
                lastMsg.thoughtProcess = [...currentThoughts];
                return newMsgs;
            });
            setCurrentThoughts([]);

        } catch (error) {
            console.error(error);
            setMessages(prev => [...prev, { role: 'assistant', content: "Error: Could not connect to API." }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <main className="flex h-screen flex-col items-center bg-white text-gray-900">
            {/* Simple Header */}
            <header className="w-full border-b border-gray-100 py-4 px-6 flex justify-between items-center bg-white z-10">
                <h1 className="text-lg font-bold tracking-tight">AURA AI</h1>
                <span className="text-xs text-gray-400 font-mono">v3.2</span>
            </header>

            {/* Chat Area */}
            <div className="flex-1 w-full max-w-3xl overflow-y-auto p-4 space-y-8">
                {messages.length === 0 && (
                    <div className="h-[50vh] flex flex-col items-center justify-center text-gray-400 space-y-4">
                        <p>감사 업무를 위한 AI 에이전트입니다. 질문을 입력해주세요.</p>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <div key={idx} className="space-y-2">
                        {/* Header for Message (Role label) */}
                        <div className="flex items-center gap-2">
                            <span className={`text-xs font-bold uppercase ${msg.role === 'user' ? 'text-gray-400' : 'text-blue-600'}`}>
                                {msg.role === 'user' ? 'You' : 'AURA'}
                            </span>
                        </div>

                        {/* Thought Process (Collapsible) */}
                        {msg.role === 'assistant' && msg.thoughtProcess && msg.thoughtProcess.length > 0 && (
                            <ThoughtProcessView steps={msg.thoughtProcess} isDone={!isLoading || idx < messages.length - 1} />
                        )}

                        {/* Content */}
                        {(msg.content || isLoading) && (
                            <div className={`text-[15px] leading-7 ${msg.role === 'user' ? 'text-gray-800' : 'text-gray-900'}`}>
                                {(
                                    msg.role === 'assistant' ? (
                                        <Typewriter text={msg.content} />
                                    ) : (
                                        <div className="whitespace-pre-wrap">{msg.content}</div>
                                    )
                                )}
                            </div>
                        )}

                        {/* Divider */}
                        {idx < messages.length - 1 && <hr className="border-gray-50 mt-6" />}
                    </div>
                ))}

                {/* Active Thoughts */}
                {isLoading && currentThoughts.length > 0 && (
                    <div className="space-y-4">
                        <FloatingDots />
                        <ThoughtProcessView steps={currentThoughts} isDone={false} />
                    </div>
                )}

                <div ref={messagesEndRef} className="h-4" />
            </div>

            {/* Input Area */}
            <div className="w-full max-w-3xl p-4">
                <form onSubmit={handleSubmit} className="relative">
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Message..."
                        className="w-full border border-gray-200 rounded-lg px-4 py-3 focus:outline-none focus:border-black transition-colors bg-gray-50 pr-12 text-sm shadow-sm"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !query.trim()}
                        className="absolute right-3 top-3 text-gray-400 hover:text-black disabled:text-gray-200 transition-colors"
                    >
                        {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                    </button>
                </form>
            </div>
        </main>
    );
}

// --- Sub-components ---

// Imports at top
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// ... (existing code)

function Typewriter({ text, onComplete }: { text: string; onComplete?: () => void }) {
    const [displayedText, setDisplayedText] = useState('');
    const indexRef = useRef(0);

    useEffect(() => {
        // Reset if text implies a new message (simple heuristic)
        if (!text) {
            setDisplayedText('');
            indexRef.current = 0;
            return;
        }

        // If current text matches target, do nothing (or handle updates)
        if (displayedText === text) return;

        // If we already displayed everything, don't restart (unless text grew?)
        if (indexRef.current >= text.length) {
            if (text.length > displayedText.length) {
                // Continue typing
            } else {
                return;
            }
        }

        const interval = setInterval(() => {
            if (indexRef.current < text.length) {
                // Determine chunk size (simulate token streaming: 2-5 chars)
                const chunkSize = Math.floor(Math.random() * 4) + 2;
                const nextIndex = Math.min(indexRef.current + chunkSize, text.length);

                const slice = text.substring(0, nextIndex); // Take substring from 0 to current
                setDisplayedText(slice);
                indexRef.current = nextIndex;
            } else {
                clearInterval(interval);
                if (onComplete) onComplete();
            }
        }, 10);

        return () => clearInterval(interval);
    }, [text]);

    return (
        <div className="markdown-prose">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {displayedText}
            </ReactMarkdown>
        </div>
    );
}

function FloatingDots() {
    return (
        <div className="flex gap-1.5 py-2 pl-1">
            <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
            <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
            <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" />
        </div>
    );
}

function ThoughtProcessView({ steps, isDone }: { steps: ThoughtStep[], isDone: boolean }) {
    const [isOpen, setIsOpen] = useState(false); // Default closed

    return (
        <div className="bg-gray-50 rounded-lg border border-gray-100 overflow-hidden">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-gray-500 hover:bg-gray-100 transition-colors"
            >
                <div className="flex items-center gap-2">
                    {isOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                    <span>{isDone ? `분석 과정 보기 (${steps.length}단계)` : '분석 진행 중...'}</span>
                </div>
            </button>

            {isOpen && (
                <div className="px-3 pb-3 space-y-1.5">
                    {steps.map((step, i) => (
                        <div key={i} className="flex gap-2 text-xs text-gray-600 pl-1">
                            <span>{step.content}</span>
                        </div>
                    ))}
                    {!isDone && <div className="pl-4 text-xs text-gray-400 italic">Thinking...</div>}
                </div>
            )}
        </div>
    );
}
