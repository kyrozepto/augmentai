import { useState } from 'react'
import { motion } from 'framer-motion'

function Curriculum() {
    const [config, setConfig] = useState({ total_epochs: 100, num_stages: 3, strategy: 'linear' })
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)

    const build = async () => {
        setLoading(true)
        try {
            const res = await fetch('/api/curriculum', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            })
            if (res.ok) setResult(await res.json())
        } catch (e) { console.error(e) }
        setLoading(false)
    }

    const getDifficultyStyles = (d) => {
        switch (d) {
            case 'easy': return { border: 'border-green-400/50', bg: 'bg-green-400/10', text: 'text-green-400/80' }
            case 'medium': return { border: 'border-yellow-400/50', bg: 'bg-yellow-400/10', text: 'text-yellow-400/80' }
            case 'hard': return { border: 'border-red-400/50', bg: 'bg-red-400/10', text: 'text-red-400/80' }
            default: return { border: 'border-white/20', bg: 'bg-white/5', text: 'text-white/60' }
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
                        CURRICULUM_MODULE
                    </span>
                </motion.div>
                <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white mb-2">
                    CURRICULUM :: BUILDER
                </h1>
                <p className="text-sm text-white/50">
                    Design adaptive augmentation schedules for progressive training
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
                        CONFIG
                    </h2>
                    <div className="space-y-5">
                        <div>
                            <label className="text-[10px] tracking-wider text-white/40 block mb-2">
                                TOTAL_EPOCHS: <span className="text-white/60">{config.total_epochs}</span>
                            </label>
                            <input
                                type="range"
                                min="50"
                                max="200"
                                value={config.total_epochs}
                                onChange={(e) => setConfig({ ...config, total_epochs: parseInt(e.target.value) })}
                                className="w-full"
                            />
                        </div>
                        <div>
                            <label className="text-[10px] tracking-wider text-white/40 block mb-2">
                                NUM_STAGES: <span className="text-white/60">{config.num_stages}</span>
                            </label>
                            <input
                                type="range"
                                min="2"
                                max="5"
                                value={config.num_stages}
                                onChange={(e) => setConfig({ ...config, num_stages: parseInt(e.target.value) })}
                                className="w-full"
                            />
                        </div>
                        <div>
                            <label className="text-[10px] tracking-wider text-white/40 block mb-2">
                                STRATEGY
                            </label>
                            <select
                                value={config.strategy}
                                onChange={(e) => setConfig({ ...config, strategy: e.target.value })}
                                className="w-full"
                            >
                                <option value="linear">Linear</option>
                                <option value="exponential">Exponential</option>
                                <option value="step">Step</option>
                            </select>
                        </div>
                        <motion.button
                            onClick={build}
                            disabled={loading}
                            whileHover={{ scale: 1.02, y: -2 }}
                            whileTap={{ scale: 0.98 }}
                            className="pixel-btn w-full"
                        >
                            {loading ? 'BUILDING...' : 'BUILD_CURRICULUM →'}
                        </motion.button>
                    </div>
                </motion.div>

                {/* Schedule */}
                <motion.div
                    initial={{ opacity: 0, x: 30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="card"
                >
                    <h2 className="text-sm font-bold tracking-[0.12em] text-white mb-6">
                        SCHEDULE
                    </h2>

                    {result ? (
                        <div className="space-y-3">
                            {result.stages.map((stage, i) => {
                                const styles = getDifficultyStyles(stage.difficulty)
                                return (
                                    <motion.div
                                        key={i}
                                        initial={{ opacity: 0, scale: 0.95, x: -20 }}
                                        animate={{ opacity: 1, scale: 1, x: 0 }}
                                        transition={{ delay: i * 0.1 }}
                                        whileHover={{ x: 4, borderColor: 'rgba(255,255,255,0.4)' }}
                                        className={`p-4 border ${styles.border} ${styles.bg} transition-colors`}
                                    >
                                        <div className="flex justify-between mb-3">
                                            <span className="font-bold text-white">Stage {i + 1}</span>
                                            <span className={`label-tag text-[9px] ${styles.text}`}>
                                                {stage.difficulty.toUpperCase()}
                                            </span>
                                        </div>
                                        <div className="text-[11px] text-white/50 space-y-2">
                                            <div className="flex items-center gap-2">
                                                <span className="text-white/30">◇</span>
                                                Epochs {stage.epoch_start} → {stage.epoch_end}
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <span className="text-white/30">◈</span>
                                                Strength: {(stage.augmentation_strength * 100).toFixed(0)}%
                                            </div>
                                            <div className="flex flex-wrap gap-1 mt-3">
                                                {stage.transforms.map(t => (
                                                    <span
                                                        key={t}
                                                        className="px-2 py-0.5 text-[9px] bg-white/10 border border-white/20 text-white/70"
                                                    >
                                                        {t}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    </motion.div>
                                )
                            })}
                        </div>
                    ) : (
                        <div className="text-center py-12">
                            <motion.div
                                className="w-16 h-16 mx-auto mb-4 text-white/15"
                                animate={{ rotate: [0, 360] }}
                                transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                            >
                                <svg viewBox="0 0 24 24" className="w-full h-full">
                                    <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="1" strokeDasharray="4 2" />
                                    <path d="M12 2 L12 12 L18 18" stroke="currentColor" strokeWidth="1" fill="none" />
                                </svg>
                            </motion.div>
                            <p className="text-sm text-white/30">
                                Configure and build to see schedule
                            </p>
                        </div>
                    )}
                </motion.div>
            </div>
        </div>
    )
}

export default Curriculum
