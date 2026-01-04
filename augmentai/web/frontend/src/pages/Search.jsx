import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

function Search() {
    const [searchId, setSearchId] = useState(null)
    const [status, setStatus] = useState('idle') // idle, running, completed
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
                <h1 className="text-2xl font-bold tracking-widest mb-2">
                    AUTOSEARCH :: OPTIMIZER
                </h1>
                <p className="text-sm opacity-70">
                    Automatically find optimal augmentation policies
                </p>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Configuration */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="card"
                >
                    <h2 className="text-sm font-bold tracking-widest mb-4">
                        SEARCH_CONFIG
                    </h2>

                    <div className="space-y-4">
                        <div>
                            <label className="text-[10px] tracking-widest block mb-1">
                                DATASET_PATH
                            </label>
                            <input
                                type="text"
                                value={config.dataset_path}
                                onChange={(e) => setConfig({ ...config, dataset_path: e.target.value })}
                                placeholder="./dataset"
                                className="w-full bg-white/50 border-2 border-black px-3 py-2 text-sm font-mono"
                                disabled={status === 'running'}
                            />
                        </div>

                        <div>
                            <label className="text-[10px] tracking-widest block mb-1">
                                DOMAIN
                            </label>
                            <select
                                value={config.domain}
                                onChange={(e) => setConfig({ ...config, domain: e.target.value })}
                                className="w-full bg-white/50 border-2 border-black px-3 py-2 text-sm font-mono"
                                disabled={status === 'running'}
                            >
                                <option value="natural">Natural Images</option>
                                <option value="medical">Medical Imaging</option>
                                <option value="ocr">OCR / Documents</option>
                                <option value="satellite">Satellite</option>
                            </select>
                        </div>

                        <div>
                            <label className="text-[10px] tracking-widest block mb-1">
                                NUM_TRIALS: {config.num_trials}
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

                        <button
                            onClick={startSearch}
                            disabled={status === 'running' || !config.dataset_path}
                            className="pixel-btn w-full"
                        >
                            {status === 'running' ? 'SEARCHING...' : 'START SEARCH'}
                        </button>
                    </div>
                </motion.div>

                {/* Progress & Results */}
                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="card"
                >
                    <h2 className="text-sm font-bold tracking-widest mb-4">
                        SEARCH_PROGRESS
                    </h2>

                    {status === 'idle' ? (
                        <p className="text-sm opacity-50 text-center py-8">
                            Configure and start a search to see progress
                        </p>
                    ) : (
                        <div className="space-y-4">
                            {/* Progress Bar */}
                            <div>
                                <div className="flex justify-between text-[10px] mb-1">
                                    <span>TRIALS: {progress.trials}/{progress.total}</span>
                                    <span>{progressPercent}%</span>
                                </div>
                                <div className="h-4 bg-black/10 border-2 border-black overflow-hidden">
                                    <motion.div
                                        className="h-full bg-black"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${progressPercent}%` }}
                                        transition={{ duration: 0.3 }}
                                    />
                                </div>
                            </div>

                            {/* Best Score */}
                            <div className="text-center py-4">
                                <span className="text-[10px] tracking-widest block">BEST_SCORE</span>
                                <span className="text-4xl font-bold">
                                    {(progress.score * 100).toFixed(1)}%
                                </span>
                            </div>

                            {/* Status */}
                            <div className="text-center">
                                <span className={`label-tag ${status === 'completed' ? 'bg-green-200' : ''}`}>
                                    {status === 'running' && '⏳ RUNNING'}
                                    {status === 'completed' && '✅ COMPLETED'}
                                </span>
                            </div>
                        </div>
                    )}
                </motion.div>
            </div>

            {/* Best Policy */}
            {bestPolicy && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="card mt-6"
                >
                    <h2 className="text-sm font-bold tracking-widest mb-4">
                        BEST_POLICY
                    </h2>
                    <pre className="bg-black text-white p-4 text-xs overflow-x-auto">
                        {JSON.stringify(bestPolicy, null, 2)}
                    </pre>
                </motion.div>
            )}
        </div>
    )
}

export default Search
