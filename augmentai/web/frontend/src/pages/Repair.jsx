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

    const getActionStyles = (action) => {
        switch (action) {
            case 'relabel': return { border: 'border-red-400/50', bg: 'bg-red-400/10', text: 'text-red-400/80' }
            case 'review': return { border: 'border-yellow-400/50', bg: 'bg-yellow-400/10', text: 'text-yellow-400/80' }
            default: return { border: 'border-green-400/50', bg: 'bg-green-400/10', text: 'text-green-400/80' }
        }
    }

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
                        REPAIR_MODULE
                    </span>
                </motion.div>
                <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white mb-2">
                    DATA :: REPAIR
                </h1>
                <p className="text-sm text-white/50">
                    Identify and fix uncertain or mislabeled samples
                </p>
            </motion.div>

            {/* Configuration */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="card mb-6"
            >
                <div className="flex flex-wrap gap-4 items-end">
                    <div className="flex-1 min-w-[200px]">
                        <label className="text-[10px] tracking-wider text-white/40 block mb-2">
                            DATASET_PATH
                        </label>
                        <input
                            type="text"
                            value={config.dataset_path}
                            onChange={(e) => setConfig({ ...config, dataset_path: e.target.value })}
                            placeholder="./dataset"
                            className="w-full"
                        />
                    </div>
                    <div className="w-48">
                        <label className="text-[10px] tracking-wider text-white/40 block mb-2">
                            THRESHOLD: <span className="text-white/60">{config.confidence_threshold}</span>
                        </label>
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
                    <motion.button
                        onClick={analyze}
                        disabled={loading || !config.dataset_path}
                        whileHover={{ scale: 1.02, y: -2 }}
                        whileTap={{ scale: 0.98 }}
                        className="pixel-btn"
                    >
                        {loading ? 'ANALYZING...' : 'ANALYZE →'}
                    </motion.button>
                </div>
            </motion.div>

            {/* Results */}
            {result && (
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="card"
                >
                    <div className="flex gap-3 mb-6">
                        <span className="label-tag">
                            <span className="text-white/40">◫</span> {result.total_samples} total
                        </span>
                        <span className="label-tag label-tag-warning">
                            <span className="text-yellow-400/60">◆</span> {result.uncertain_count} uncertain
                        </span>
                    </div>

                    <div className="space-y-2">
                        {result.samples.map((s, i) => {
                            const styles = getActionStyles(s.action)
                            return (
                                <motion.div
                                    key={s.id}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: i * 0.05 }}
                                    whileHover={{ x: 4 }}
                                    className={`border-l-4 ${styles.border} ${styles.bg} p-4 flex justify-between items-center`}
                                >
                                    <div>
                                        <span className="font-bold text-sm text-white">{s.id}</span>
                                        <span className="text-[10px] text-white/40 ml-3 font-mono">{s.path}</span>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <span className="text-sm text-white/60">
                                            {(s.confidence * 100).toFixed(0)}%
                                        </span>
                                        <span className={`label-tag text-[9px] ${styles.text}`}>
                                            {s.action.toUpperCase()}
                                        </span>
                                    </div>
                                </motion.div>
                            )
                        })}
                    </div>
                </motion.div>
            )}
        </div>
    )
}

export default Repair
