'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Send, ChevronDown, ChevronRight, Loader2, BookOpen, RefreshCw, X, AlertTriangle, Edit3, Save, FileText, Copy, Check, Maximize2, Minimize2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

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

// Markdown Custom Components for Report Styling
const MarkdownComponents = {
    h1: ({ node, ...props }: any) => <h1 className="text-xl font-bold text-gray-900 mt-6 mb-3 border-b pb-2" {...props} />,
    h2: ({ node, ...props }: any) => <h2 className="text-lg font-bold text-gray-800 mt-5 mb-2" {...props} />,
    h3: ({ node, ...props }: any) => <h3 className="text-md font-bold text-gray-800 mt-4 mb-2" {...props} />,
    p: ({ node, ...props }: any) => <p className="text-[15px] text-gray-700 leading-relaxed mb-3" {...props} />,
    ul: ({ node, ...props }: any) => <ul className="list-disc pl-5 mb-3 text-[15px] text-gray-700" {...props} />,
    ol: ({ node, ...props }: any) => <ol className="list-decimal pl-5 mb-3 text-[15px] text-gray-700" {...props} />,
    li: ({ node, ...props }: any) => <li className="mb-1" {...props} />,
    table: ({ node, ...props }: any) => <table className="min-w-full divide-y divide-gray-300 border border-gray-200 mb-4" {...props} />,
    th: ({ node, ...props }: any) => <th className="bg-gray-50 px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b" {...props} />,
    td: ({ node, ...props }: any) => <td className="px-3 py-2 whitespace-pre-wrap text-[15px] text-gray-600 border-b" {...props} />,
    blockquote: ({ node, ...props }: any) => <blockquote className="border-l-4 border-indigo-500 pl-4 italic text-gray-600 my-4" {...props} />,
};

const PLACEHOLDER_MAP: Record<string, string> = {
    "사건 제목": "예: OO공사 공공기관 채용 비리 의혹",
    "감사 배경": "예: 내부 제보 접수로 인한 특정 감사 착수",
    "감사 목적": "예: 채용 절차의 공정성 검증 및 위반 사항 적발",
    "감사 방법": "예: 관련 서류 검토 및 관계자 대면 조사",
    "감사 기간": "예: 2023.11.01 ~ 2023.11.15",
    "대상 기관": "예: 한국철도공사",
    "문제점": "예: 채용 점수 조작 및 서류 위조 정황 발견",
};

export default function Home() {
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [currentThoughts, setCurrentThoughts] = useState<ThoughtStep[]>([]);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // [Fix] Ref to track latest messages for async callbacks (avoiding stale closures)
    const messagesRef = useRef<Message[]>([]);
    useEffect(() => {
        messagesRef.current = messages;
    }, [messages]);

    // [Report State]
    const [showReport, setShowReport] = useState(false);
    const [reportContent, setReportContent] = useState("");
    const [isReportLoading, setIsReportLoading] = useState(false);
    // New States for Report Enhancement
    const [reportState, setReportState] = useState<'idle' | 'checking' | 'missing_info' | 'generating' | 'done'>('idle');
    const [missingFields, setMissingFields] = useState<string[]>([]);
    const [userInputs, setUserInputs] = useState<Record<string, string>>({});
    const [isEditing, setIsEditing] = useState(false);
    const [editContent, setEditContent] = useState("");
    const [isCopied, setIsCopied] = useState(false);
    const [isInputFullscreen, setIsInputFullscreen] = useState(false);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // [New] Unique Session ID per page refresh
    const sessionId = useRef<string>('');
    if (!sessionId.current) {
        sessionId.current = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    }

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(scrollToBottom, [messages, currentThoughts]);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
        }
    }, [query]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e as any);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim() || isLoading) return;

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
                    history: history,
                    session_id: currentSessionId,
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
                                    newMsgs[newMsgs.length - 1] = { ...assistantMsg, thoughtProcess: [...currentThoughts] };
                                    newMsgs[newMsgs.length - 1] = { ...assistantMsg, thoughtProcess: [...currentThoughts] };
                                    return newMsgs;
                                });
                            } else if (data.type === 'command') {
                                // [New] Handle Commands (e.g., Open Report)
                                if (data.content === 'open_report') {
                                    if (!showReport) setShowReport(true);
                                    // Optionally trigger check immediately
                                    setTimeout(() => checkReadiness(), 500);
                                }
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

    // [Report Generation Flow]
    const checkReadiness = async () => {
        const currentMessages = messagesRef.current; // Use Ref
        if (currentMessages.length === 0) return;

        setReportState('checking');
        setUserInputs({}); // Reset previous inputs if any

        try {
            const res = await fetch('http://localhost:8000/check_report_readiness', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: "Check Readiness",
                    history: currentMessages.map(m => ({ role: m.role, content: m.content })),
                    session_id: sessionId.current,
                }),
            });
            const data = await res.json();

            if (data.status === 'missing_info') {
                setMissingFields(data.missing_fields || []);
                setReportState('missing_info');
            } else {
                // Ready, proceed to generate
                generateReport({});
            }
        } catch (error) {
            console.error("Readiness Check Error:", error);
            // Fallback to direct generation attempt
            generateReport({});
        }
    };

    const generateReport = async (additionalInfo: Record<string, string>) => {
        setReportState('generating');
        const currentMessages = messagesRef.current; // Use Ref
        // Keep UI loading state sync

        try {
            const res = await fetch('http://localhost:8000/generate_report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: "Generate Report",
                    history: currentMessages.map(m => ({ role: m.role, content: m.content })),
                    session_id: sessionId.current,
                    additional_info: additionalInfo
                }),
            });
            const data = await res.json();
            setReportContent(data.report);
            setEditContent(data.report);
            setReportState('done');
            setIsEditing(false);

        } catch (error) {
            console.error("Report Gen Error:", error);
            alert("보고서 생성 중 오류가 발생했습니다.");
            setReportState('idle');
        }
    };

    const handleInputChange = (field: string, value: string) => {
        setUserInputs(prev => ({ ...prev, [field]: value }));
    };

    const handleCopy = async () => {
        if (!reportContent) return;
        try {
            await navigator.clipboard.writeText(reportContent);
            setIsCopied(true);
            setTimeout(() => setIsCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy', err);
        }
    };

    // Wrapper to open panel and start check
    const startReportProcess = () => {
        if (!showReport) setShowReport(true);
        if (reportContent) {
            // If already has content, do nothing unless user explicitly refreshes?
            // For now, if state is idle/done, checking is optional but good for "Refresh"
        }
    };


    return (
        <main className="flex h-screen flex-col items-center bg-white text-gray-900 overflow-hidden">
            {/* Header */}
            <header className="w-full border-b border-gray-100 py-4 px-6 flex justify-between items-center bg-white z-10 shrink-0">
                <div className="flex items-center gap-4">
                    <h1 className="text-lg font-bold tracking-tight">AURA AI</h1>
                    <span className="text-xs text-gray-400 font-mono">v3.3</span>
                </div>

                {/* Toggle Report Button */}
                <button
                    onClick={() => setShowReport(!showReport)}
                    className={`p-2 rounded-md transition-colors ${showReport ? 'bg-blue-50 text-blue-600' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-50'}`}
                    title={showReport ? "보고서 닫기" : "보고서 열기"}
                >
                    <BookOpen className="w-5 h-5" />
                </button>
            </header>

            <div className="flex-1 w-full flex overflow-hidden">
                {/* Left: Chat Area */}
                <div className={`flex flex-col h-full transition-all duration-300 ${showReport ? 'w-1/2 border-r border-gray-200' : 'w-full max-w-3xl mx-auto'}`}>
                    <div className="flex-1 overflow-y-auto p-4 space-y-8">
                        {messages.length === 0 && (
                            <div className="h-[50vh] flex flex-col items-center justify-center text-gray-400 space-y-4">
                                <p>감사 업무를 위한 AI 에이전트입니다. 질문을 입력해주세요.</p>
                            </div>
                        )}

                        {messages.map((msg, idx) => (
                            <div key={idx} className="space-y-2">
                                <div className="flex items-center gap-2">
                                    <span className={`text-xs font-bold uppercase ${msg.role === 'user' ? 'text-gray-400' : 'text-blue-600'}`}>
                                        {msg.role === 'user' ? 'You' : 'AURA'}
                                    </span>
                                </div>

                                {msg.role === 'assistant' && msg.thoughtProcess && msg.thoughtProcess.length > 0 && (
                                    <ThoughtProcessView steps={msg.thoughtProcess} isDone={!isLoading || idx < messages.length - 1} />
                                )}

                                {(msg.content || isLoading) && (
                                    <div className={`text-[15px] leading-7 ${msg.role === 'user' ? 'text-gray-800' : 'text-gray-900'}`}>
                                        {(msg.role === 'assistant' ? <Typewriter text={msg.content} /> : <div className="whitespace-pre-wrap">{msg.content}</div>)}
                                    </div>
                                )}

                                {idx < messages.length - 1 && <hr className="border-gray-50 mt-6" />}
                            </div>
                        ))}

                        {isLoading && currentThoughts.length > 0 && (
                            <div className="space-y-4">
                                <FloatingDots />
                                <ThoughtProcessView steps={currentThoughts} isDone={false} />
                            </div>
                        )}
                        <div ref={messagesEndRef} className="h-4" />
                    </div>

                    {/* Input Area */}
                    <div className={`transition-all duration-300 ${isInputFullscreen ? 'fixed inset-0 z-50 bg-white/80 backdrop-blur-sm flex justify-center pt-24 px-4' : 'p-4 bg-white border-t border-gray-100'}`}>

                        <form onSubmit={handleSubmit} className={`relative transition-all duration-300 ${isInputFullscreen ? 'w-full max-w-3xl h-[600px]' : 'w-full'}`}>
                            {isInputFullscreen && (
                                <button
                                    type="button"
                                    onClick={() => setIsInputFullscreen(false)}
                                    className="absolute -top-10 right-0 p-2 text-gray-500 hover:text-gray-800 bg-white rounded-full shadow-sm hover:shadow-md transition-all"
                                    title="Close Fullscreen"
                                >
                                    <Minimize2 className="w-5 h-5" />
                                </button>
                            )}

                            <div className={`relative w-full border border-gray-200 rounded-2xl bg-gray-50 transition-all overflow-hidden flex items-center ${isInputFullscreen ? 'h-full shadow-2xl bg-white border-gray-300' : 'shadow-sm'}`}>
                                <textarea
                                    ref={textareaRef}
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    placeholder="Message..."
                                    className={`w-full bg-transparent px-4 py-3.5 pr-12 text-[15px] focus:outline-none resize-none ${isInputFullscreen ? 'h-full p-8 text-lg leading-relaxed' : 'max-h-[200px] overflow-y-auto'}`}
                                    rows={1}
                                    disabled={isLoading}
                                    style={{ minHeight: isInputFullscreen ? '100%' : '52px' }}
                                />
                                <div className={`absolute right-2 flex gap-1 ${isInputFullscreen ? 'bottom-4 right-4' : 'bottom-2'}`}>
                                    {(query.split('\n').length > 3 || isInputFullscreen) && !isInputFullscreen && (
                                        <button
                                            type="button"
                                            onClick={() => setIsInputFullscreen(true)}
                                            className="p-1.5 text-gray-400 hover:text-black transition-colors rounded hover:bg-gray-200"
                                            title="Full Screen Input"
                                        >
                                            <Maximize2 className="w-4 h-4" />
                                        </button>
                                    )}
                                    <button
                                        type="submit"
                                        disabled={isLoading || !query.trim()}
                                        className="p-2 bg-black text-white rounded-full hover:bg-gray-800 disabled:bg-gray-200 transition-colors shadow-sm"
                                    >
                                        {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                                    </button>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>

                {/* Right: Report Panel */}
                {showReport && (
                    <div className="w-1/2 h-full bg-white flex flex-col animate-in slide-in-from-right duration-300">
                        <div className="flex-none p-4 border-b border-gray-100 flex justify-between items-center bg-gray-50/50">
                            <h2 className="font-semibold text-gray-800 flex items-center gap-2 text-sm">
                                📄 Audit Report
                                {isReportLoading && <span className="text-xs font-normal text-blue-600 animate-pulse">Running...</span>}
                            </h2>
                            <div className="flex gap-2">
                                {/* Edit Toggle */}
                                {reportState === 'done' && (
                                    <button
                                        onClick={() => setIsEditing(!isEditing)}
                                        className={`p-1.5 rounded hover:bg-white transition-all ${isEditing ? 'text-blue-600 bg-white' : 'text-gray-500 hover:text-blue-600'}`}
                                        title={isEditing ? "View Mode" : "Edit Mode"}
                                    >
                                        {isEditing ? <FileText className="w-4 h-4" /> : <Edit3 className="w-4 h-4" />}
                                    </button>
                                )}
                                {/* Copy Button */}
                                {(reportState === 'done' || (reportState === 'idle' && reportContent)) && (
                                    <button
                                        onClick={handleCopy}
                                        className={`p-1.5 rounded hover:bg-white transition-all ${isCopied ? 'text-green-600' : 'text-gray-500 hover:text-blue-600'}`}
                                        title="Copy to Clipboard"
                                    >
                                        {isCopied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                                    </button>
                                )}
                                <button
                                    onClick={checkReadiness}
                                    title="Regenerate Report"
                                    disabled={reportState === 'generating' || reportState === 'checking' || messages.length === 0}
                                    className={`p-1.5 rounded hover:bg-white text-gray-500 hover:text-blue-600 transition-all ${reportState === 'generating' || reportState === 'checking' ? 'animate-spin' : ''}`}
                                >
                                    <RefreshCw className="w-4 h-4" />
                                </button>
                                <button onClick={() => setShowReport(false)} className="p-1.5 hover:bg-white rounded text-gray-400 hover:text-gray-600 transition-all">
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                        <div className="flex-1 overflow-y-auto p-8 bg-white">
                            {reportState === 'idle' && !reportContent && (
                                <div className="flex flex-col items-center justify-center py-20 text-gray-400 space-y-3">
                                    <p className="text-sm">대화 내용을 바탕으로 보고서를 작성합니다.</p>
                                    <button
                                        onClick={checkReadiness}
                                        disabled={messages.length === 0}
                                        className="text-xs bg-blue-50 text-blue-600 px-3 py-1.5 rounded-full hover:bg-blue-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        보고서 작성 시작
                                    </button>
                                </div>
                            )}

                            {reportState === 'checking' && (
                                <div className="flex flex-col items-center justify-center py-20 text-gray-500 space-y-4">
                                    <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
                                    <p className="text-sm">필수 정보를 확인하고 있습니다...</p>
                                </div>
                            )}

                            {reportState === 'missing_info' && (
                                <div className="max-w-md mx-auto animate-in slide-in-from-bottom-5 duration-300">
                                    <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6">
                                        <div className="flex items-start gap-3">
                                            <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                                            <div>
                                                <h3 className="text-sm font-bold text-amber-800">추가 정보 필요</h3>
                                                <p className="text-xs text-amber-700 mt-1">완성도 높은 보고서를 위해 다음 정보를 입력해주세요.</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="space-y-4">
                                        {missingFields.map((field, idx) => (
                                            <div key={idx}>
                                                <label className="block text-xs font-semibold text-gray-700 mb-1">{field}</label>
                                                <input
                                                    type="text"
                                                    value={userInputs[field] || ''}
                                                    onChange={(e) => handleInputChange(field, e.target.value)}
                                                    className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
                                                    placeholder={PLACEHOLDER_MAP[field] || `${field} 입력...`}
                                                />
                                            </div>
                                        ))}
                                        <button
                                            onClick={() => generateReport(userInputs)}
                                            className="w-full bg-blue-600 text-white rounded-lg py-2.5 text-sm font-semibold hover:bg-blue-700 transition-colors mt-4"
                                        >
                                            입력 완료 및 보고서 생성
                                        </button>
                                    </div>
                                </div>
                            )}

                            {reportState === 'generating' && (
                                <div className="flex flex-col items-center justify-center py-20 text-gray-500 space-y-4">
                                    <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
                                    <TypingAnimation text="보고서 초안을 작성 중입니다..." />
                                </div>
                            )}

                            {(reportState === 'done' || (reportState === 'idle' && reportContent)) && (
                                <>
                                    {isEditing ? (
                                        <div className="h-full flex flex-col">
                                            <textarea
                                                value={editContent}
                                                onChange={(e) => setEditContent(e.target.value)}
                                                className="flex-1 w-full border border-gray-200 rounded-lg p-4 font-mono text-sm leading-6 focus:outline-none focus:border-blue-500 resize-none"
                                            />
                                            <div className="flex justify-end pt-3">
                                                <button
                                                    onClick={() => {
                                                        setReportContent(editContent);
                                                        setIsEditing(false);
                                                    }}
                                                    className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
                                                >
                                                    <Save className="w-3.5 h-3.5" /> 반영 완료
                                                </button>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="markdown-prose space-y-4">
                                            <ReactMarkdown remarkPlugins={[remarkGfm]} components={MarkdownComponents}>{reportContent}</ReactMarkdown>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </main>
    );
}

// --- Sub-components (Unchanged logic, minor tweaks) ---

function TypingAnimation({ text }: { text: string }) {
    const [dots, setDots] = useState('');
    useEffect(() => {
        const interval = setInterval(() => {
            setDots(prev => prev.length >= 3 ? '' : prev + '.');
        }, 500);
        return () => clearInterval(interval);
    }, []);
    return <p className="text-sm">{text}{dots}</p>;
}


function Typewriter({ text, onComplete }: { text: string; onComplete?: () => void }) {
    const [displayedText, setDisplayedText] = useState('');
    const indexRef = useRef(0);

    useEffect(() => {
        if (!text) { setDisplayedText(''); indexRef.current = 0; return; }
        if (displayedText === text) return;

        // Fast forward if text grew significantly (paste or big chunk)
        if (text.length - indexRef.current > 50) {
            setDisplayedText(text);
            indexRef.current = text.length;
            return;
        }

        const interval = setInterval(() => {
            if (indexRef.current < text.length) {
                const chunkSize = Math.floor(Math.random() * 3) + 1; // Slower, more natural typing
                const nextIndex = Math.min(indexRef.current + chunkSize, text.length);
                setDisplayedText(text.substring(0, nextIndex));
                indexRef.current = nextIndex;
            } else {
                clearInterval(interval);
                if (onComplete) onComplete();
            }
        }, 15);
        return () => clearInterval(interval);
    }, [text]);

    return (
        <div className="markdown-prose">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={MarkdownComponents}>
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
    const [isOpen, setIsOpen] = useState(false);
    return (
        <div className="bg-gray-50 rounded-lg border border-gray-100 overflow-hidden max-w-2xl">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-gray-500 hover:bg-gray-100 transition-colors"
            >
                <div className="flex items-center gap-2">
                    {isOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                    <span>{isDone ? `Context Process (${steps.length})` : 'Processing...'}</span>
                </div>
            </button>
            {isOpen && (
                <div className="px-3 pb-3 space-y-1.5">
                    {steps.map((step, i) => (
                        <div key={i} className="flex gap-2 text-xs text-gray-600 pl-1 border-l-2 border-gray-200">
                            <div className="font-mono text-[10px] text-blue-500 shrink-0 w-16 truncate text-right mr-1">{step.node}</div>
                            <div>{step.content}</div>
                        </div>
                    ))}
                    {!isDone && <div className="pl-4 text-xs text-gray-400 italic">Computing...</div>}
                </div>
            )}
        </div>
    );
}
