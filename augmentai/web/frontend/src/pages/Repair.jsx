import { useState } from 'react'
import { motion } from 'framer-motion'

function Repair() {
    const [config, setConfig] = useState({ dataset_path: '', confidence_threshold: 0.5 })
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)

    const analyze = async () => {
        setLoading(true)
        try {
            const res = await fetch('/api/repair', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            })
            if (res.ok) setResult(await res.json())
        } catch (e) { console.error(e) }
        setLoading(false)
    }

    const getActionColor = (action) => {
        switch (action) {
            case 'relabel': return 'bg-red-200 border-red-600'
            case 'review': return 'bg-yellow-200 border-yellow-600'
            default: return 'bg-green-200 border-green-600'
        }
    }

    return (
        <div className="max-w-6xl mx-auto">
            <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
                <h1 className="text-2xl font-bold tracking-widest mb-2">DATA :: REPAIR</h1>
                <p className="text-sm opacity-70">Identify and fix uncertain or mislabeled samples</p>
            </motion.div>

            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="card mb-6">
                <div className="flex gap-4 items-end">
                    <div className="flex-1">
                        <label className="text-[10px] tracking-widest block mb-1">DATASET_PATH</label>
                        <input
                            type="text"
                            value={config.dataset_path}
                            onChange={(e) => setConfig({ ...config, dataset_path: e.target.value })}
                            placeholder="./dataset"
                            className="w-full bg-white/50 border-2 border-black px-3 py-2 text-sm"
                        />
                    </div>
                    <div className="w-40">
                        <label className="text-[10px] tracking-widest block mb-1">THRESHOLD: {config.confidence_threshold}</label>
                        <input
                            type="range"
                            min="0.1"
                            max="0.9"
                            step="0.1"
                            value={config.confidence_threshold}
                            onChange={(e) => setConfig({ ...config, confidence_threshold: parseFloat(e.target.value) })}
                            className="w-full"
                        />
                    </div>
                    <button onClick={analyze} disabled={loading} className="pixel-btn">
                        {loading ? '...' : 'ANALYZE'}
                    </button>
                </div>
            </motion.div>

            {result && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="card">
                    <div className="flex gap-4 mb-4">
                        <span className="label-tag">📊 {result.total_samples} total</span>
                        <span className="label-tag bg-yellow-200">⚠️ {result.uncertain_count} uncertain</span>
                    </div>
                    <div className="space-y-2">
                        {result.samples.map((s, i) => (
                            <motion.div
                                key={s.id}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: i * 0.05 }}
                                className={`border-l-4 p-3 flex justify-between items-center ${getActionColor(s.action)}`}
                            >
                                <div>
                                    <span className="font-bold text-sm">{s.id}</span>
                                    <span className="text-[10px] opacity-70 ml-2">{s.path}</span>
                                </div>
                                <div className="flex items-center gap-4">
                                    <span className="text-sm">{(s.confidence * 100).toFixed(0)}%</span>
                                    <span className="label-tag text-[9px]">{s.action.toUpperCase()}</span>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </motion.div>
            )}
        </div>
    )
}

export default Repair
