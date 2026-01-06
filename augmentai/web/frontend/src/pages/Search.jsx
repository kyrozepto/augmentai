import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

function Search() {
    const [searchId, setSearchId] = useState(null)
    const [status, setStatus] = useState('idle')
    const [progress, setProgress] = useState({ trials: 0, total: 20, score: 0 })
    const [bestPolicy, setBestPolicy] = useState(null)
    const [config, setConfig] = useState({
        dataset_path: '',
        domain: 'natural',
        num_trials: 20,
    })
    const wsRef = useRef(null)

    const startSearch = async () => {
        try {
            const res = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            })

            if (res.ok) {
                const data = await res.json()
                setSearchId(data.id)
                setStatus('running')
                connectWebSocket(data.id)
            }
        } catch (err) {
            console.error(err)
        }
    }

    const connectWebSocket = (id) => {
        const ws = new WebSocket(`ws://localhost:8000/api/search/${id}/ws`)
        wsRef.current = ws

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data)
            setProgress({
                trials: data.trials_completed,
                total: data.total_trials,
                score: data.best_score || 0,
            })
            if (data.best_policy) {
                setBestPolicy(data.best_policy)
            }
            if (data.status === 'completed') {
                setStatus('completed')
            }
        }

        ws.onclose = () => {
            wsRef.current = null
        }
    }

    useEffect(() => {
        return () => {
            if (wsRef.current) {
                wsRef.current.close()
            }
        }
    }, [])

    const progressPercent = progress.total > 0
        ? Math.round((progress.trials / progress.total) * 100)
        : 0

    return (
        <div className="max-w-6xl mx-auto">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8"
            >
                <motion.div
                    className="inline-block mb-3"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                >
                    <span className="text-[10px] tracking-[0.2em] text-white/40 border border-white/20 px-3 py-1">
                        OPTIMIZER_MODULE
                    </span>
                </motion.div>
                <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white mb-2">
                    AUTOSEARCH :: OPTIMIZER
                </h1>
                <p className="text-sm text-white/50">
                    Automatically find optimal augmentation policies via evolutionary search
                </p>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Configuration */}
                <motion.div
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="card"
                >
                    <h2 className="text-sm font-bold tracking-[0.12em] text-white mb-6">
                        SEARCH_CONFIG
                    </h2>

                    <div className="space-y-5">
                        <div>
                            <label className="text-[10px] tracking-wider text-white/40 block mb-2">
                                DATASET_PATH
                            </label>
                            <input
                                type="text"
                                value={config.dataset_path}
                                onChange={(e) => setConfig({ ...config, dataset_path: e.target.value })}
                                placeholder="./dataset"
                                className="w-full"
                                disabled={status === 'running'}
                            />
                        </div>

                        <div>
                            <label className="text-[10px] tracking-wider text-white/40 block mb-2">
                                DOMAIN
                            </label>
                            <select
                                value={config.domain}
                                onChange={(e) => setConfig({ ...config, domain: e.target.value })}
                                className="w-full"
                                disabled={status === 'running'}
                            >
                                <option value="natural">Natural Images</option>
                                <option value="medical">Medical Imaging</option>
                                <option value="ocr">OCR / Documents</option>
                                <option value="satellite">Satellite</option>
                            </select>
                        </div>

                        <div>
                            <label className="text-[10px] tracking-wider text-white/40 block mb-2">
                                NUM_TRIALS: <span className="text-white/60">{config.num_trials}</span>
                            </label>
                            <input
                                type="range"
                                min="5"
                                max="50"
                                value={config.num_trials}
                                onChange={(e) => setConfig({ ...config, num_trials: parseInt(e.target.value) })}
                                className="w-full"
                                disabled={status === 'running'}
                            />
                        </div>

                        <motion.button
                            onClick={startSearch}
                            disabled={status === 'running' || !config.dataset_path}
                            whileHover={{ scale: 1.02, y: -2 }}
                            whileTap={{ scale: 0.98 }}
                            className="pixel-btn w-full"
                        >
                            {status === 'running' ? 'SEARCHING...' : 'START_SEARCH →'}
                        </motion.button>
                    </div>
                </motion.div>

                {/* Progress & Results */}
                <motion.div
                    initial={{ opacity: 0, x: 30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="card"
                >
                    <h2 className="text-sm font-bold tracking-[0.12em] text-white mb-6">
                        SEARCH_PROGRESS
                    </h2>

                    {status === 'idle' ? (
                        <div className="text-center py-12">
                            <motion.div
                                className="w-16 h-16 mx-auto mb-4 text-white/15"
                                animate={{ rotate: [0, 360] }}
                                transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                            >
                                <svg viewBox="0 0 24 24" className="w-full h-full">
                                    <path d="M2 20 L6 14 L10 16 L14 8 L18 10 L22 4" stroke="currentColor" strokeWidth="1.5" fill="none" />
                                    <circle cx="14" cy="8" r="2" fill="currentColor" />
                                </svg>
                            </motion.div>
                            <p className="text-sm text-white/30">
                                Configure and start a search
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {/* Progress Bar */}
                            <div>
                                <div className="flex justify-between text-[10px] mb-2 text-white/50">
                                    <span>TRIALS: {progress.trials}/{progress.total}</span>
                                    <span>{progressPercent}%</span>
                                </div>
                                <div className="progress-bar">
                                    <motion.div
                                        className="progress-bar-fill"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${progressPercent}%` }}
                                        transition={{ duration: 0.3 }}
                                    />
                                </div>
                            </div>

                            {/* Best Score */}
                            <div className="text-center py-6 border border-white/10 bg-white/5">
                                <span className="text-[10px] tracking-wider text-white/40 block mb-2">
                                    BEST_SCORE
                                </span>
                                <motion.span
                                    className="text-5xl font-bold text-white"
                                    key={progress.score}
                                    initial={{ scale: 1.1 }}
                                    animate={{ scale: 1 }}
                                >
                                    {(progress.score * 100).toFixed(1)}%
                                </motion.span>
                            </div>

                            {/* Status */}
                            <div className="text-center">
                                <span className={`label-tag ${status === 'completed' ? 'label-tag-success' : ''}`}>
                                    <span className={`status-dot ${status === 'running' ? 'pending' : 'online'} mr-2`} />
                                    {status === 'running' && 'RUNNING'}
                                    {status === 'completed' && 'COMPLETED'}
                                </span>
                            </div>
                        </div>
                    )}
                </motion.div>
            </div>

            {/* Best Policy */}
            {bestPolicy && (
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="card mt-6"
                >
                    <h2 className="text-sm font-bold tracking-[0.12em] text-white mb-4">
                        BEST_POLICY
                    </h2>
                    <pre className="code-block overflow-x-auto">
                        {JSON.stringify(bestPolicy, null, 2)}
                    </pre>
                </motion.div>
            )}
        </div>
    )
}

export default Search
