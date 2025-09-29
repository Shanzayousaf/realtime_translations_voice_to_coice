import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Mic, MicOff, Volume2, VolumeX, Settings, Wifi, WifiOff } from 'lucide-react'

const App = () => {
  // State management
  const [isRecording, setIsRecording] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [sourceLang, setSourceLang] = useState('en')
  const [targetLang, setTargetLang] = useState('fr')
  const [sourceText, setSourceText] = useState('')
  const [targetText, setTargetText] = useState('')
  const [error, setError] = useState('')
  const [metrics, setMetrics] = useState({
    latency: 0,
    audioLevel: 0,
    chunksProcessed: 0
  })

  // Refs
  const mediaRecorderRef = useRef(null)
  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const asrWebSocketRef = useRef(null)
  const ttsWebSocketRef = useRef(null)
  const audioQueueRef = useRef([])
  const isPlayingRef = useRef(false)
  const animationFrameRef = useRef(null)

  // WebSocket URLs from environment
  const ASR_WS_URL = import.meta.env.VITE_ASR_URL || 'ws://localhost:8000'
  const TTS_WS_URL = import.meta.env.VITE_TTS_URL || 'ws://localhost:8002'

  // Audio processing utilities
  const resampleAudio = useCallback((audioBuffer, targetSampleRate = 16000) => {
    const sourceSampleRate = audioBuffer.sampleRate
    if (sourceSampleRate === targetSampleRate) {
      return audioBuffer
    }

    const ratio = sourceSampleRate / targetSampleRate
    const newLength = Math.floor(audioBuffer.length / ratio)
    const newBuffer = new AudioBuffer({
      numberOfChannels: 1,
      length: newLength,
      sampleRate: targetSampleRate
    })

    const sourceData = audioBuffer.getChannelData(0)
    const targetData = newBuffer.getChannelData(0)

    for (let i = 0; i < newLength; i++) {
      const sourceIndex = Math.floor(i * ratio)
      targetData[i] = sourceData[sourceIndex]
    }

    return newBuffer
  }, [])

  const audioBufferToPCM16 = useCallback((audioBuffer) => {
    const length = audioBuffer.length
    const buffer = new ArrayBuffer(length * 2)
    const view = new DataView(buffer)
    const data = audioBuffer.getChannelData(0)

    for (let i = 0; i < length; i++) {
      const sample = Math.max(-1, Math.min(1, data[i]))
      view.setInt16(i * 2, sample * 0x7FFF, true)
    }

    return buffer
  }, [])

  // WebSocket connections
  const connectASR = useCallback(() => {
    try {
      const ws = new WebSocket(`${ASR_WS_URL}/ws?channelId=main&src=${sourceLang}&tgt=${targetLang}`)
      
      ws.onopen = () => {
        console.log('ASR WebSocket connected')
        setIsConnected(true)
        setError('')
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'asr_partial' || data.type === 'asr_final') {
            setSourceText(data.text)
            if (data.type === 'asr_final') {
              console.log('Final transcription:', data.text)
            }
          }
        } catch (err) {
          console.error('Error parsing ASR message:', err)
        }
      }

      ws.onclose = () => {
        console.log('ASR WebSocket disconnected')
        setIsConnected(false)
      }

      ws.onerror = (error) => {
        console.error('ASR WebSocket error:', error)
        setError('Connection error. Please check if services are running.')
      }

      asrWebSocketRef.current = ws
    } catch (err) {
      console.error('Failed to connect ASR WebSocket:', err)
      setError('Failed to connect to speech recognition service.')
    }
  }, [ASR_WS_URL, sourceLang, targetLang])

  const connectTTS = useCallback(() => {
    try {
      const ws = new WebSocket(`${TTS_WS_URL}/ws?channelId=main`)
      
      ws.onopen = () => {
        console.log('TTS WebSocket connected')
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'audio_chunk') {
            // Decode base64 audio and add to queue
            const audioData = atob(data.pcm16_base64)
            const audioBuffer = new ArrayBuffer(audioData.length)
            const view = new Uint8Array(audioBuffer)
            for (let i = 0; i < audioData.length; i++) {
              view[i] = audioData.charCodeAt(i)
            }
            audioQueueRef.current.push(audioBuffer)
            playNextAudioChunk()
          } else if (data.type === 'audio_final') {
            console.log('Audio synthesis completed')
          }
        } catch (err) {
          console.error('Error parsing TTS message:', err)
        }
      }

      ws.onclose = () => {
        console.log('TTS WebSocket disconnected')
      }

      ws.onerror = (error) => {
        console.error('TTS WebSocket error:', error)
      }

      ttsWebSocketRef.current = ws
    } catch (err) {
      console.error('Failed to connect TTS WebSocket:', err)
    }
  }, [TTS_WS_URL])

  // Audio playback
  const playNextAudioChunk = useCallback(() => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) return

    isPlayingRef.current = true
    const audioData = audioQueueRef.current.shift()

    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)()
      const audioBuffer = audioContext.createBuffer(1, audioData.byteLength / 2, 22050)
      const channelData = audioBuffer.getChannelData(0)
      const view = new DataView(audioData)

      for (let i = 0; i < audioData.byteLength / 2; i++) {
        const sample = view.getInt16(i * 2, true) / 32768.0
        channelData[i] = sample
      }

      const source = audioContext.createBufferSource()
      source.buffer = audioBuffer
      source.connect(audioContext.destination)
      source.onended = () => {
        isPlayingRef.current = false
        playNextAudioChunk() // Play next chunk if available
      }
      source.start()
    } catch (err) {
      console.error('Audio playback error:', err)
      isPlayingRef.current = false
    }
  }, [])

  // Audio visualization
  const updateAudioVisualizer = useCallback(() => {
    if (!analyserRef.current || !isRecording) return

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(dataArray)
    
    const average = dataArray.reduce((a, b) => a + b) / dataArray.length
    setMetrics(prev => ({ ...prev, audioLevel: average }))

    animationFrameRef.current = requestAnimationFrame(updateAudioVisualizer)
  }, [isRecording])

  // Start/stop recording
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      })

      // Set up audio context for visualization
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)()
      const source = audioContextRef.current.createMediaStreamSource(stream)
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 256
      source.connect(analyserRef.current)

      // Set up media recorder
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })

      mediaRecorderRef.current.ondataavailable = async (event) => {
        if (event.data.size > 0 && asrWebSocketRef.current?.readyState === WebSocket.OPEN) {
          try {
            // Convert to PCM16 and send
            const arrayBuffer = await event.data.arrayBuffer()
            const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer)
            const resampledBuffer = resampleAudio(audioBuffer)
            const pcm16Data = audioBufferToPCM16(resampledBuffer)
            
            asrWebSocketRef.current.send(pcm16Data)
            setMetrics(prev => ({ ...prev, chunksProcessed: prev.chunksProcessed + 1 }))
          } catch (err) {
            console.error('Error processing audio chunk:', err)
          }
        }
      }

      mediaRecorderRef.current.start(200) // Send chunks every 200ms
      setIsRecording(true)
      updateAudioVisualizer()
      
    } catch (err) {
      console.error('Error starting recording:', err)
      setError('Failed to access microphone. Please check permissions.')
    }
  }, [resampleAudio, audioBufferToPCM16, updateAudioVisualizer])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close()
    }
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }
    
    setIsRecording(false)
    setMetrics(prev => ({ ...prev, audioLevel: 0 }))
  }, [])

  // Send translation request to TTS
  const sendToTTS = useCallback(async (text) => {
    if (ttsWebSocketRef.current?.readyState === WebSocket.OPEN) {
      const request = {
        text: text,
        sequence: Date.now(),
        language: targetLang,
        final: true
      }
      ttsWebSocketRef.current.send(JSON.stringify(request))
    }
  }, [targetLang])

  // Handle translation updates
  useEffect(() => {
    if (sourceText && sourceText !== targetText) {
      // Simulate translation (in real app, this would come from MT service)
      setTargetText(`[Translated: ${sourceText}]`)
      sendToTTS(`[Translated: ${sourceText}]`)
    }
  }, [sourceText, targetText, sendToTTS])

  // Initialize connections
  useEffect(() => {
    connectASR()
    connectTTS()
    
    return () => {
      if (asrWebSocketRef.current) {
        asrWebSocketRef.current.close()
      }
      if (ttsWebSocketRef.current) {
        ttsWebSocketRef.current.close()
      }
      stopRecording()
    }
  }, [connectASR, connectTTS, stopRecording])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [])

  return (
    <div className="app">
      <h1>🎤 Real-time Voice Translation</h1>
      
      <div className="controls">
        <div className="language-selector">
          <label>
            From:
            <select 
              value={sourceLang} 
              onChange={(e) => setSourceLang(e.target.value)}
              disabled={isRecording}
            >
              <option value="en">English</option>
              <option value="fr">French</option>
              <option value="es">Spanish</option>
              <option value="de">German</option>
              <option value="it">Italian</option>
            </select>
          </label>
          
          <label>
            To:
            <select 
              value={targetLang} 
              onChange={(e) => setTargetLang(e.target.value)}
              disabled={isRecording}
            >
              <option value="fr">French</option>
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="de">German</option>
              <option value="it">Italian</option>
            </select>
          </label>
        </div>

        <button
          className={`control-button ${isRecording ? 'recording' : ''}`}
          onClick={isRecording ? stopRecording : startRecording}
          disabled={!isConnected}
        >
          {isRecording ? (
            <>
              <MicOff size={20} style={{ marginRight: '8px' }} />
              Stop Recording
            </>
          ) : (
            <>
              <Mic size={20} style={{ marginRight: '8px' }} />
              Start Recording
            </>
          )}
        </button>
      </div>

      <div className={`status ${isConnected ? 'connected' : 'disconnected'} ${isRecording ? 'recording' : ''}`}>
        {isConnected ? (
          <>
            <Wifi size={16} style={{ marginRight: '8px' }} />
            {isRecording ? 'Recording...' : 'Connected'}
          </>
        ) : (
          <>
            <WifiOff size={16} style={{ marginRight: '8px' }} />
            Disconnected
          </>
        )}
      </div>

      {error && (
        <div className="error">
          {error}
        </div>
      )}

      <div className="transcript-container">
        <div className="transcript-box">
          <h3>🎯 Source ({sourceLang.toUpperCase()})</h3>
          <div className={`transcript-text ${!sourceText ? 'empty' : ''}`}>
            {sourceText || 'Start speaking to see transcription...'}
          </div>
        </div>

        <div className="transcript-box">
          <h3>🌍 Translation ({targetLang.toUpperCase()})</h3>
          <div className={`transcript-text ${!targetText ? 'empty' : ''}`}>
            {targetText || 'Translation will appear here...'}
          </div>
        </div>
      </div>

      <div className="audio-visualizer">
        {isRecording && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            {Array.from({ length: 20 }, (_, i) => (
              <div
                key={i}
                className="visualizer-bar"
                style={{
                  height: `${10 + (metrics.audioLevel / 255) * 30}px`,
                  animationDelay: `${i * 0.05}s`
                }}
              />
            ))}
          </div>
        )}
      </div>

      <div className="metrics">
        <div className="metric">
          <div className="metric-label">Latency</div>
          <div className="metric-value">{metrics.latency}ms</div>
        </div>
        <div className="metric">
          <div className="metric-label">Audio Level</div>
          <div className="metric-value">{Math.round(metrics.audioLevel)}</div>
        </div>
        <div className="metric">
          <div className="metric-label">Chunks Processed</div>
          <div className="metric-value">{metrics.chunksProcessed}</div>
        </div>
        <div className="metric">
          <div className="metric-label">Status</div>
          <div className="metric-value">
            {isRecording ? 'Recording' : isConnected ? 'Ready' : 'Disconnected'}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
