import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowRight,
  BarChart3,
  BrainCircuit,
  BriefcaseBusiness,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Clock3,
  GraduationCap,
  MessageSquareText,
  RotateCcw,
  Send,
  ShieldCheck,
  Sparkles,
  Target,
  Zap,
} from 'lucide-react';

const API = 'http://127.0.0.1:5000/api';
const types = ['General', 'Technical', 'Behavioral', 'HR'];
const roleOptions = [
  'Product Manager',
  'Product Designer',
  'Frontend Developer',
  'Backend Developer',
  'Full Stack Developer',
  'Software Engineer',
  'Data Analyst',
  'Data Scientist',
  'UX Designer',
  'QA Engineer',
  'DevOps Engineer',
  'Business Analyst',
  'Project Manager',
  'Customer Success Manager',
  'Sales Manager',
  'Marketing Manager',
  'HR Manager',
  'Operations Manager',
  'Machine Learning Engineer',
  'Cybersecurity Analyst',
  'AI Engineer',
  'Data Engineer',
  'Platform Engineer',
  'Technical Product Manager',
  'Account Manager',
  'Recruiter',
  'Financial Analyst',
  'Operations Analyst',
  'Research Analyst',
  'Solution Architect',
];
const defaultRole = 'Product Manager';
const evaluationDimensions = [
  'technicalAccuracy',
  'conceptualUnderstanding',
  'completeness',
  'practicalExamples',
  'clarity',
  'reasoning',
];

function roundHalfToEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (Math.abs(fraction - 0.5) < Number.EPSILON) {
    return lower % 2 === 0 ? lower : lower + 1;
  }
  return Math.round(value);
}

function isValidEvaluation(value) {
  if (!value || typeof value !== 'object') return false;
  if (!evaluationDimensions.every((name) => Number.isInteger(value.dimensionScores?.[name])
    && value.dimensionScores[name] >= 0
    && value.dimensionScores[name] <= 100)) return false;
  if (Object.keys(value.dimensionScores).length !== evaluationDimensions.length) return false;

  const lists = ['correctConcepts', 'missingConcepts', 'incorrectConcepts', 'actionableFeedback'];
  if (!lists.every((name) => Array.isArray(value[name])
    && value[name].every((item) => typeof item === 'string' && item.trim()))) return false;
  if (!value.actionableFeedback.length) return false;
  if (!Array.isArray(value.conceptReview) || !value.conceptReview.every((item) => (
    item
    && typeof item.concept === 'string'
    && ['covered', 'missing', 'incorrect'].includes(item.status)
    && typeof item.evidence === 'string'
  ))) return false;

  const matchingConcepts = (status, listName) => {
    const reviewed = value.conceptReview
      .filter((item) => item.status === status)
      .map((item) => item.concept)
      .sort();
    return JSON.stringify(reviewed) === JSON.stringify([...value[listName]].sort());
  };
  if (!matchingConcepts('covered', 'correctConcepts')
    || !matchingConcepts('missing', 'missingConcepts')
    || !matchingConcepts('incorrect', 'incorrectConcepts')) return false;

  const scores = evaluationDimensions.map((name) => value.dimensionScores[name]);
  const expectedOverall = roundHalfToEven(scores.reduce((sum, score) => sum + score, 0) / scores.length);
  if (!Number.isInteger(value.overallScore) || value.overallScore !== expectedOverall) return false;
  if (!['incorrect', 'partially_correct', 'correct'].includes(value.assessment)) return false;
  if (typeof value.explanation !== 'string' || value.explanation.trim().length < 12) return false;
  if (typeof value.feedback !== 'string' || value.feedback.trim().length < 12) return false;
  if (typeof value.improvedAnswer !== 'string' || value.improvedAnswer.trim().length < 20) return false;

  const expectedAssessment = value.overallScore >= 60
    && value.missingConcepts.length === 0
    && value.incorrectConcepts.length === 0
    ? 'correct'
    : value.overallScore < 40 && value.correctConcepts.length === 0
      ? 'incorrect'
      : 'partially_correct';
  return value.assessment === expectedAssessment;
}

function App() {
  const [role, setRole] = useState(defaultRole);
  const [experience, setExperience] = useState('Entry level');
  const [interviewType, setInterviewType] = useState('Technical');
  const [difficulty, setDifficulty] = useState('Medium');
  const [questionCount, setQuestionCount] = useState('5');
  const [stage, setStage] = useState('setup');
  const [userName, setUserName] = useState('Candidate');
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [answers, setAnswers] = useState([]);
  const [questionNumber, setQuestionNumber] = useState(0);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [expectedConcepts, setExpectedConcepts] = useState([]);
  const [usedQuestions, setUsedQuestions] = useState([]);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState(null);
  const [readyForNext, setReadyForNext] = useState(false);
  const [reportError, setReportError] = useState('');
  const [pdfBusy, setPdfBusy] = useState(false);
  const [pdfError, setPdfError] = useState('');
  const [speechError, setSpeechError] = useState('');
  const [speechTranscript, setSpeechTranscript] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [mode, setMode] = useState('offline');
  const [apiStatus, setApiStatus] = useState('Checking coach…');
  const [history, setHistory] = useState([]);
  const endRef = useRef(null);
  const recognitionRef = useRef(null);
  const [interviewDate, setInterviewDate] = useState('');

  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem('interviewiq-history') || '[]');
    setHistory(Array.isArray(saved) ? saved : []);

    async function checkStatus() {
      try {
        const response = await fetch(`${API}/health`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Backend health check failed.');
        if (!data.connected) {
          setMode('offline');
          setApiStatus(`Ollama offline · ${data.error || 'start Ollama to continue'}`);
        } else if (!data.modelAvailable) {
          setMode('offline');
          setApiStatus(`Model unavailable · ${data.model}`);
        } else {
          setMode('ai');
          setApiStatus(`Ollama connected · ${data.model}`);
        }
      } catch {
        setMode('offline');
        setApiStatus('Backend unavailable · check port 5000');
      }
    }

    checkStatus();
    const timer = window.setInterval(checkStatus, 15000);
    return () => {
      window.clearInterval(timer);
      recognitionRef.current?.stop();
    };
  }, []);

  useEffect(() => {
    localStorage.setItem('interviewiq-history', JSON.stringify(history));
  }, [history]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, busy]);

  const totalQuestions = Number(questionCount) || 5;

  const payload = (action, extra = {}) => ({
    action,
    role,
    experience,
    interviewType,
    difficulty,
    questionNumber,
    answers,
    usedQuestions,
    currentQuestion,
    ...extra,
  });

  async function callCoach(body) {
    try {
      const res = await fetch(`${API}/interview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.error || 'The coach could not complete that request.');
      }

      const data = await res.json();
      if (body.action === 'start' || body.action === 'next') {
        if (typeof data.question !== 'string' || !Array.isArray(data.expectedConcepts) || !data.expectedConcepts.length) {
          throw new Error('The coach returned an invalid question. Please retry.');
        }
      } else if (body.action === 'evaluate') {
        if (!isValidEvaluation(data.evaluation)) {
          throw new Error('The coach returned an invalid evaluation. Your answer is still available to retry.');
        }
      } else if (body.action === 'finish' && (!Number.isFinite(data.score) || !Array.isArray(data.answers))) {
        throw new Error('The coach returned an invalid performance report.');
      }
      return data;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'The backend is unavailable.';
      throw new Error(message.includes('fetch') || message.includes('Failed') ? 'Cannot reach the coach. Start the Python backend on port 5000.' : message);
    }
  }

  async function startInterview() {
    if (!role.trim()) return;
    const candidateName = userName.trim() || 'Candidate';
    setUserName(candidateName);
    setBusy(true);
    setMessages([]);
    setAnswers([]);
    setQuestionNumber(0);
    setCurrentQuestion('');
    setExpectedConcepts([]);
    setUsedQuestions([]);
    setReport(null);
    setReportError('');
    setReadyForNext(false);
    setInterviewDate(new Date().toISOString());
    setStage('interview');

    try {
      const data = await callCoach(payload('start'));
      setMode(data.mode === 'ollama' ? 'ai' : data.mode || 'offline');
      setMessages([
        {
          id: Date.now(),
          role: 'coach',
          text: data.message,
          question: data.question,
          feedback: data.feedback || '',
          hint: data.hint || '',
        },
      ]);
      setCurrentQuestion(data.question || '');
      setExpectedConcepts(data.expectedConcepts || []);
      setUsedQuestions((prev) => (data.question ? [...prev, data.question] : prev));
    } catch (error) {
      setMessages([{ id: Date.now(), role: 'coach', text: error.message }]);
      setStage('setup');
    } finally {
      setBusy(false);
    }
  }

  async function finalizeInterview(nextAnswers) {
    setBusy(true);
    setReportError('');
    try {
      const data = await callCoach(payload('finish', { answers: nextAnswers }));
      setReport(data);
      setMode(data.mode === 'ollama' ? 'ai' : data.mode || mode);
      setStage('report');
      const score = Number(data.score || 0);
      setHistory((prev) => [
        {
          role,
          interviewType,
          difficulty,
          score,
          date: new Date().toLocaleString(),
        },
        ...prev,
      ].slice(0, 6));
    } catch (error) {
      setReportError(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function sendAnswer() {
    const answer = draft.trim();
    if (!answer || busy || readyForNext) return;
    setBusy(true);

    try {
      const data = await callCoach(
        payload('evaluate', {
          answer,
          question: currentQuestion,
          expectedConcepts,
        })
      );
      const answerEntry = { question: currentQuestion, answer, evaluation: data.evaluation };
      const nextAnswers = [...answers, answerEntry];
      setAnswers(nextAnswers);
      setDraft('');
      setSpeechTranscript('');
      setQuestionNumber(nextAnswers.length);
      setMessages((prev) => [
        ...prev,
        { id: Date.now(), role: 'user', text: answer },
        { id: Date.now() + 1, role: 'coach', text: data.evaluation.feedback, evaluation: data.evaluation },
      ]);
      if (nextAnswers.length >= totalQuestions) {
        await finalizeInterview(nextAnswers);
      } else {
        setReadyForNext(true);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 1, role: 'coach', text: `${error.message} Your answer has been kept; retry the evaluation when ready.` },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function goToNextQuestion() {
    if (busy || !readyForNext) return;
    setBusy(true);
    try {
      const data = await callCoach(payload('next', {
        questionNumber,
        usedQuestions,
        answers,
      }));
      setCurrentQuestion(data.question);
      setExpectedConcepts(data.expectedConcepts);
      setUsedQuestions((prev) => [...prev, data.question]);
      setReadyForNext(false);
      setMessages((prev) => [...prev, {
        id: Date.now(),
        role: 'coach',
        text: data.message,
        question: data.question,
      }]);
    } catch (error) {
      setMessages((prev) => [...prev, {
        id: Date.now(),
        role: 'coach',
        text: `${error.message} Your evaluated answer is saved; retry to get the next question.`,
      }]);
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(event) {
    event.preventDefault();
    if (readyForNext) {
      goToNextQuestion();
    } else {
      sendAnswer();
    }
  }

  function toggleSpeechRecognition() {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
      return;
    }
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      setSpeechError('Speech recognition is not supported in this browser. You can still type your answer.');
      return;
    }

    setSpeechError('');
    setSpeechTranscript('');
    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = navigator.language || 'en-US';
    recognition.onresult = (event) => {
      let interim = '';
      let completed = '';
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const phrase = event.results[index][0].transcript;
        if (event.results[index].isFinal) completed += `${phrase} `;
        else interim += phrase;
      }
      if (completed) {
        const finalText = completed.trim();
        setDraft((current) => [current.trim(), finalText].filter(Boolean).join(' '));
      }
      setSpeechTranscript(`${completed}${interim}`.trim());
    };
    recognition.onerror = (event) => {
      const messages = {
        'not-allowed': 'Microphone permission was denied. Allow microphone access or type your answer.',
        'service-not-allowed': 'Speech recognition is blocked by the browser. You can type your answer instead.',
        'no-speech': 'No speech was detected. Try again or type your answer.',
        'audio-capture': 'No microphone was found. Connect a microphone or type your answer.',
        network: 'Speech recognition could not connect. Check your connection or type your answer.',
      };
      setSpeechError(messages[event.error] || `Speech recognition failed (${event.error}). You can type your answer.`);
      setIsListening(false);
    };
    recognition.onend = () => setIsListening(false);
    recognitionRef.current = recognition;
    try {
      recognition.start();
      setIsListening(true);
    } catch {
      setSpeechError('The microphone could not start. Check its permission and try again.');
      setIsListening(false);
    }
  }

  async function downloadReport() {
    setPdfBusy(true);
    setPdfError('');
    try {
      const response = await fetch(`${API}/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          candidate: userName,
          role,
          experience,
          interviewType,
          difficulty,
          interviewDate,
          report,
          answers,
        }),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'The PDF report could not be generated.');
      }
      const blob = await response.blob();
      if (blob.type !== 'application/pdf') throw new Error('The server did not return a PDF report.');
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'interview-performance-report.pdf';
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => window.URL.revokeObjectURL(url), 1000);
    } catch (error) {
      setPdfError(error instanceof Error ? error.message : 'The PDF report could not be downloaded.');
    } finally {
      setPdfBusy(false);
    }
  }

  function openReport() {
    setStage('report');
  }

  function reset() {
    setStage('setup');
    setMessages([]);
    setAnswers([]);
    setQuestionNumber(0);
    setCurrentQuestion('');
    setExpectedConcepts([]);
    setUsedQuestions([]);
    setDraft('');
    setReadyForNext(false);
    setReportError('');
    setPdfError('');
    setSpeechError('');
    setSpeechTranscript('');
    setInterviewDate('');
    recognitionRef.current?.stop();
    setIsListening(false);
  }

  const progressValue = useMemo(
    () => Math.min(100, ((questionNumber + 1) / totalQuestions) * 100),
    [questionNumber, totalQuestions]
  );

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <BrainCircuit size={23} />
          </div>
          <div>
            <b>
              interview<span>IQ</span>
            </b>
            <small>AI CAREER STUDIO</small>
          </div>
        </div>

        <div className="side-label">WORKSPACE</div>
        <button className={`nav-item ${stage === 'setup' ? 'active' : ''}`} onClick={reset}>
          <Sparkles size={17} /> Interview studio
        </button>
        <button className={`nav-item ${stage === 'report' ? 'active' : ''}`} onClick={openReport}>
          <BarChart3 size={17} /> Performance report
        </button>

        <div className="sidebar-bottom">
          <div className="upgrade-card">
            <div className="upgrade-icon">
              <GraduationCap size={18} />
            </div>
            <b>Practice makes progress.</b>
            <p>Build confidence one answer at a time.</p>
            <div className="mini-progress">
              <span />
            </div>
            <small>YOUR NEXT LEVEL STARTS HERE</small>
          </div>

          <button type="button" className="profile" onClick={openReport} style={{ border: 'none', background: 'transparent', width: '100%', padding: 0, textAlign: 'left', color: 'inherit' }}>
            <div className="avatar">{(userName || 'C').charAt(0).toUpperCase()}</div>
            <div>
              <b>{userName || 'Candidate'}</b>
              <small>{role} · report</small>
            </div>
            <span className="online-dot" />
          </button>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={14} />{' '}
            <strong>{stage === 'setup' ? 'Interview studio' : stage === 'interview' ? 'Live interview' : 'Your report'}</strong>
          </div>

          <div className="top-actions">
            <span className={`status-pill ${mode === 'ai' ? 'ai' : 'offline'}`}>
              <span className="status-dot" />
              {apiStatus}
            </span>
            <div className="help">
              <CircleHelp size={17} />
              <span>Help center</span>
            </div>
          </div>
        </header>

        {stage === 'setup' && (
          <section className="setup-page">
            <div className="hero">
              <div className="eyebrow">
                <span className="eyebrow-line" /> YOUR PERSONAL INTERVIEW PARTNER
              </div>
              <h1>
                Turn interview nerves
                <br />
                into <span>career confidence.</span>
              </h1>
              <p>
                Practice realistic interviews, get thoughtful coaching, and walk into your next opportunity prepared.
              </p>

              <div className="hero-tags">
                <span>
                  <ShieldCheck size={14} /> Private practice
                </span>
                <span>
                  <Zap size={14} /> Instant coaching
                </span>
                <span>
                  <Target size={14} /> Role-focused
                </span>
              </div>

              <div className="hero-art">
                <div className="orb orb-one" />
                <div className="orb orb-two" />
                <div className="orbit orbit-one" />
                <div className="orbit orbit-two" />
                <div className="hero-brain">
                  <BrainCircuit size={54} />
                </div>

                <div className="float-card float-score">
                  <div className="float-icon">
                    <Target size={16} />
                  </div>
                  <div>
                    <b>Skill growth</b>
                    <small>Every session counts</small>
                  </div>
                  <span className="up-arrow">↗</span>
                </div>

                <div className="float-card float-feedback">
                  <div className="check-bubble">
                    <CheckCircle2 size={17} />
                  </div>
                  <div>
                    <b>Actionable feedback</b>
                    <small>Made for your next step</small>
                  </div>
                </div>

                <div className="sparkle s1">✦</div>
                <div className="sparkle s2">✧</div>
              </div>
            </div>

            <div className="section-head">
              <div>
                <span className="section-kicker">LET'S GET YOU READY</span>
                <h2>Build your interview session</h2>
                <p>Customize the experience to match your next big opportunity.</p>
              </div>
              <div className="step-count">
                <span>01</span> / 03
              </div>
            </div>

            <div className="setup-grid">
              <div className="setup-card">
                <div className="card-heading">
                  <div className="heading-icon purple">
                    <BriefcaseBusiness size={18} />
                  </div>
                  <div>
                    <h3>Role & experience</h3>
                    <p>We'll tailor questions to your goals.</p>
                  </div>
                </div>

                <label>Your name</label>
                <input
                  type="text"
                  value={userName}
                  onChange={(e) => setUserName(e.target.value)}
                  placeholder="Candidate"
                  style={{ width: '100%', marginBottom: '14px', padding: '10px 12px', borderRadius: '10px', border: '1px solid #3b3d55', background: '#0f1320', color: '#edf3ff' }}
                />

                <label>Target job title</label>
                <select value={role} onChange={(e) => setRole(e.target.value)}>
                  {roleOptions.map((title) => (
                    <option key={title} value={title}>
                      {title}
                    </option>
                  ))}
                </select>

                <label>Experience level</label>
                <div className="choice-row">
                  {['Student / Fresher', 'Entry level', 'Mid-level', 'Senior'].map((level) => (
                    <button
                      key={level}
                      type="button"
                      className={`choice ${experience === level ? 'selected' : ''}`}
                      onClick={() => setExperience(level)}
                    >
                      {level}
                    </button>
                  ))}
                </div>

                <div className="hint">
                  <Sparkles size={14} /> Your questions adapt to your experience.
                </div>
              </div>

              <div className="setup-card">
                <div className="card-heading">
                  <div className="heading-icon blue">
                    <MessageSquareText size={18} />
                  </div>
                  <div>
                    <h3>Interview style</h3>
                    <p>Choose the kind of conversation to practice.</p>
                  </div>
                </div>

                <label>Interview focus</label>
                <div className="type-grid">
                  {types.map((type, idx) => (
                    <button
                      key={type}
                      type="button"
                      className={`type-option ${interviewType === type ? 'selected' : ''}`}
                      onClick={() => setInterviewType(type)}
                    >
                      <span className="type-icon">
                        {idx === 0 && <MessageSquareText size={17} />}
                        {idx === 1 && <BrainCircuit size={17} />}
                        {idx === 2 && <Target size={17} />}
                        {idx === 3 && <BriefcaseBusiness size={17} />}
                      </span>

                      <span>
                        <b>{type}</b>
                        <small>
                          {['Mixed questions', 'Skills & problem solving', 'Real-world examples', 'Career & culture'][idx]}
                        </small>
                      </span>

                      {interviewType === type && <CheckCircle2 className="type-check" size={16} />}
                    </button>
                  ))}
                </div>

                <div className="settings-row">
                  <div>
                    <label>Difficulty</label>
                    <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
                      <option>Easy</option>
                      <option>Medium</option>
                      <option>Hard</option>
                    </select>
                  </div>

                  <div>
                    <label>Questions</label>
                    <select value={questionCount} onChange={(e) => setQuestionCount(e.target.value)}>
                      <option value="3">3 questions</option>
                      <option value="5">5 questions</option>
                      <option value="7">7 questions</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>

            <div className="start-row">
              <div className="time-estimate">
                <Clock3 size={17} />
                <span>
                  <b>
                    About {Number(questionCount) * 2}–{Number(questionCount) * 3} minutes
                  </b>
                  <small>Take your time. This is your space to grow.</small>
                </span>
              </div>

              <button type="button" className="primary-btn" onClick={startInterview} disabled={!role.trim() || busy}>
                Start mock interview <ArrowRight size={17} />
              </button>
            </div>

            <div className="feature-strip">
              <div>
                <div className="feature-icon">
                  <BrainCircuit size={19} />
                </div>
                <span>
                  <b>Smart questions</b>
                  <small>Role-aware practice</small>
                </span>
              </div>

              <div>
                <div className="feature-icon">
                  <MessageSquareText size={19} />
                </div>
                <span>
                  <b>Helpful feedback</b>
                  <small>Clear next steps</small>
                </span>
              </div>

              <div>
                <div className="feature-icon">
                  <BarChart3 size={19} />
                </div>
                <span>
                  <b>Progress report</b>
                  <small>Know what to improve</small>
                </span>
              </div>
            </div>
          </section>
        )}

        {stage === 'interview' && (
          <section className="interview-page">
            <div className="interview-header">
              <div>
                <span className="section-kicker">LIVE PRACTICE SESSION</span>
                <h1>Your interview, your progress.</h1>
                <p>
                  {role} <span className="bullet">•</span> {interviewType} interview <span className="bullet">•</span> {difficulty}
                </p>
              </div>

              <button type="button" className="quiet-btn" onClick={reset}>
                <RotateCcw size={15} /> End session
              </button>
            </div>

            <div className="session-progress">
              <div>
                <span>SESSION PROGRESS</span>
                <b>
                  {Math.min(questionNumber + 1, totalQuestions)} <small>of {totalQuestions} questions</small>
                </b>
              </div>
              <div className="progress-track">
                <span style={{ width: `${progressValue}%` }} />
              </div>
            </div>

            <div className="chat-panel">
              <div className="chat-top">
                <div className="coach-avatar">
                  <BrainCircuit size={20} />
                </div>
                <div>
                  <b>Your AI interviewer</b>
                  <small>
                    <span className="status-dot" /> {mode === 'ai' ? 'AI-powered coaching' : 'Practice coach'} · Ready to listen
                  </small>
                </div>
                <span className="live-label">LIVE SESSION</span>
              </div>

              <div className="chat-body">
                {messages.map((message) => (
                  <div className={`message-row ${message.role}`} key={message.id}>
                    {message.role === 'coach' && (
                      <div className="message-avatar">
                        <Sparkles size={15} />
                      </div>
                    )}

                    <div className="bubble">
                      <div className="message-text">{message.text}</div>

                      {(message.feedback || message.hint) && (
                        <div className="feedback-note">
                          <Sparkles size={14} />
                          <span>
                            <b>{message.feedback ? 'Coach’s note' : 'Hint'}</b>
                            <br />
                            {message.feedback || message.hint}
                          </span>
                        </div>
                      )}

                      {message.evaluation && (
                        <div className="evaluation-card">
                          <div className="evaluation-score">
                            <b>{message.evaluation.overallScore}/100</b>
                            <span>{message.evaluation.assessment.replaceAll('_', ' ')}</span>
                          </div>
                          <p>{message.evaluation.explanation}</p>
                          <div className="dimension-scores">
                            {Object.entries(message.evaluation.dimensionScores).map(([key, score]) => (
                              <div key={key} className="dimension-score">
                                <span>{{
                                  technicalAccuracy: 'Technical accuracy',
                                  conceptualUnderstanding: 'Conceptual understanding',
                                  completeness: 'Completeness',
                                  practicalExamples: 'Practical examples',
                                  clarity: 'Clarity',
                                  reasoning: 'Problem-solving',
                                }[key] || key}</span>
                                <b>{score}</b>
                                <div><i style={{ width: `${score}%` }} /></div>
                              </div>
                            ))}
                          </div>
                          {[
                            ['Correct concepts', message.evaluation.correctConcepts],
                            ['Missing concepts', message.evaluation.missingConcepts],
                            ['Incorrect concepts', message.evaluation.incorrectConcepts],
                            ['Actionable feedback', message.evaluation.actionableFeedback],
                          ].map(([label, items]) => (
                            <div className="evaluation-detail" key={label}>
                              <b>{label}</b>
                              <span>{items.length ? items.join(' · ') : 'None identified'}</span>
                            </div>
                          ))}
                          <div className="evaluation-detail improved-answer">
                            <b>Suggested improved answer</b>
                            <span>{message.evaluation.improvedAnswer}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {busy && (
                  <div className="message-row coach">
                    <div className="message-avatar">
                      <Sparkles size={15} />
                    </div>
                    <div className="bubble typing">
                      <i />
                      <i />
                      <i />
                      <span>Thinking through your response…</span>
                    </div>
                  </div>
                )}

                <div ref={endRef} />
              </div>

              <form className="answer-box" onSubmit={onSubmit}>
                <div className="answer-label">
                  <span>YOUR RESPONSE</span>
                  <span>{draft.length} characters</span>
                </div>

                <textarea
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  placeholder="Take a breath and answer as you would in a real interview…"
                  rows={4}
                  disabled={busy || readyForNext}
                />

                <div className="speech-controls">
                  <button type="button" className={`quiet-btn ${isListening ? 'recording' : ''}`} onClick={toggleSpeechRecognition} disabled={busy || readyForNext}>
                    {isListening ? 'Stop recording' : 'Use microphone'}
                  </button>
                  {speechTranscript && <span className="speech-transcript">Recognized: {speechTranscript}</span>}
                </div>
                {speechError && <p className="inline-error" role="status">{speechError}</p>}

                <div className="answer-footer">
                  <span>
                    {readyForNext
                      ? 'Answer evaluated and saved.'
                      : <><ShieldCheck size={14} /> Your practice space is for learning.</>}
                  </span>
                  <button type="submit" className="primary-btn" disabled={busy || (!readyForNext && !draft.trim())}>
                    {readyForNext ? <ArrowRight size={15} /> : <Send size={15} />}
                    {readyForNext ? 'Next question' : 'Evaluate answer'}
                  </button>
                </div>
                {reportError && (
                  <div className="retry-report">
                    <p className="inline-error" role="alert">{reportError}</p>
                    <button type="button" className="quiet-btn" onClick={() => finalizeInterview(answers)} disabled={busy}>
                      Retry final report
                    </button>
                  </div>
                )}
              </form>
            </div>

            <div className="interview-tip">
              <Sparkles size={17} />
              <span>
                <b>Quick tip</b> — Use a specific example and explain the impact of your actions. For behavioral questions,
                try Situation, Task, Action, Result.
              </span>
            </div>
          </section>
        )}

        {stage === 'report' && (
          <section className="report-page">
            {!report ? (
              <div className="report-card" style={{ marginTop: '24px' }}>
                <div className="report-card-title">
                  <div className="report-icon purple">
                    <BarChart3 size={18} />
                  </div>
                  <h3>Performance report</h3>
                </div>
                <p style={{ color: '#c7d5f3', lineHeight: 1.7, margin: 0 }}>
                  Complete a mock interview to unlock your personalized performance report, strengths, and next steps.
                </p>
                <div className="report-bottom" style={{ marginTop: '18px' }}>
                  <button type="button" className="primary-btn" onClick={() => setStage('setup')}>
                    <ArrowRight size={15} /> Start a session
                  </button>
                  <button type="button" className="quiet-btn" onClick={openReport}>
                    <BarChart3 size={15} /> View report status
                  </button>
                </div>
              </div>
            ) : (
              <>
            <div className="report-hero">
              <div>
                <span className="section-kicker">SESSION COMPLETE</span>
                <h1>You showed up. That’s progress.</h1>
                <p>{userName} · {role} · {interviewDate ? new Date(interviewDate).toLocaleDateString() : ''}</p>
              </div>

              <div className="score-ring" style={{ '--score': `${report.score || 0}%` }}>
                <div>
                  <b>{report.score || 0}</b>
                  <small>/ 100</small>
                </div>
              </div>
            </div>

            <div className="report-headline">
              <Sparkles size={19} />
              <span>{report.headline}</span>
            </div>

            <div className="report-columns">
              <div className="report-card">
                <div className="report-card-title">
                  <div className="report-icon green">
                    <CheckCircle2 size={18} />
                  </div>
                  <h3>What’s working</h3>
                </div>
                <ul>
                  {(report.strengths || []).map((item, index) => (
                    <li key={`${item}-${index}`}>{item}</li>
                  ))}
                </ul>
              </div>

              <div className="report-card">
                <div className="report-card-title">
                  <div className="report-icon orange">
                    <Target size={18} />
                  </div>
                  <h3>Room to improve</h3>
                </div>
                <ul>
                  {(report.weaknesses || report.improvements || []).map((item, index) => (
                    <li key={`${item}-${index}`}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="report-card question-report">
              <div className="report-card-title">
                <div className="report-icon blue"><MessageSquareText size={18} /></div>
                <h3>Question-by-question evaluation</h3>
              </div>
              {(report.answers || answers).map((entry, index) => (
                <article className="report-answer" key={`${index}-${entry.question}`}>
                  <h4>Question {index + 1}: {entry.question}</h4>
                  <p><b>Your answer:</b> {entry.answer}</p>
                  <p><b>Assessment:</b> {entry.evaluation.assessment.replaceAll('_', ' ')} · {entry.evaluation.overallScore}/100</p>
                  <p>{entry.evaluation.explanation}</p>
                  <div className="dimension-scores report-dimensions">
                    {Object.entries(entry.evaluation.dimensionScores).map(([key, score]) => (
                      <div className="dimension-score" key={key}>
                        <span>{{
                          technicalAccuracy: 'Technical accuracy',
                          conceptualUnderstanding: 'Conceptual understanding',
                          completeness: 'Completeness',
                          practicalExamples: 'Practical examples',
                          clarity: 'Clarity',
                          reasoning: 'Problem-solving',
                        }[key] || key}</span>
                        <b>{score}</b>
                        <div><i style={{ width: `${score}%` }} /></div>
                      </div>
                    ))}
                  </div>
                  <p><b>Correct:</b> {entry.evaluation.correctConcepts.join(' · ') || 'None identified'}</p>
                  <p><b>Missing:</b> {entry.evaluation.missingConcepts.join(' · ') || 'None identified'}</p>
                  <p><b>Incorrect:</b> {entry.evaluation.incorrectConcepts.join(' · ') || 'None identified'}</p>
                  <p><b>Improved answer:</b> {entry.evaluation.improvedAnswer}</p>
                </article>
              ))}
            </div>

            <div className="report-card next-steps">
              <div className="report-card-title">
                <div className="report-icon purple">
                  <GraduationCap size={18} />
                </div>
                <h3>Your next steps</h3>
              </div>

              <div className="next-step-list">
                {(report.nextSteps || []).map((item, index) => (
                  <div key={`${item}-${index}`}>
                    <span>{String(index + 1).padStart(2, '0')}</span>
                    {item}
                  </div>
                ))}
              </div>
            </div>

            <div className="report-bottom">
              <button type="button" className="quiet-btn" onClick={reset}>
                <ArrowRight size={15} /> Back to studio
              </button>

              <button type="button" className="quiet-btn" onClick={downloadReport} disabled={pdfBusy}>
                {pdfBusy ? 'Preparing PDF…' : 'Download PDF report'}
              </button>

              <button type="button" className="primary-btn" onClick={startInterview}>
                <RotateCcw size={16} /> Practice again
              </button>
            </div>
            {pdfError && <p className="inline-error" role="alert">{pdfError}</p>}
              </>
            )}
          </section>
        )}

        {history.length > 0 && stage === 'setup' && (
          <section className="history-panel">
            <div className="section-head compact">
              <div>
                <span className="section-kicker">RECENT HISTORY</span>
                <h2>Latest sessions</h2>
              </div>
            </div>

            <div className="history-list">
              {history.map((item, index) => (
                <div key={`${item.date}-${index}`} className="history-item">
                  <div>
                    <b>{item.role}</b>
                    <small>
                      {item.interviewType} · {item.difficulty}
                    </small>
                  </div>
                  <span>{item.score}/100</span>
                  <small>{item.date}</small>
                </div>
              ))}
            </div>
          </section>
        )}

        <footer className="footer">
          <span>© 2026 interviewIQ</span>
          <span>Designed for growth, powered by practice.</span>
          <span>
            <ShieldCheck size={13} /> Your future is worth preparing for.
          </span>
        </footer>
      </main>
    </div>
  );
}

export default App;
