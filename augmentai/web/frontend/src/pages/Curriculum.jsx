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

    const getDifficultyColor = (d) => {
        switch (d) {
            case 'easy': return 'bg-green-200'
            case 'medium': return 'bg-yellow-200'
            case 'hard': return 'bg-red-200'
            default: return 'bg-gray-200'
        }
    }

    return (
        <div className="max-w-6xl mx-auto">
            <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
                <h1 className="text-2xl font-bold tracking-widest mb-2">CURRICULUM :: BUILDER</h1>
                <p className="text-sm opacity-70">Design adaptive augmentation schedules</p>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="card">
                    <h2 className="text-sm font-bold tracking-widest mb-4">CONFIG</h2>
                    <div className="space-y-4">
                        <div>
                            <label className="text-[10px] tracking-widest block mb-1">TOTAL_EPOCHS: {config.total_epochs}</label>
                            <input type="range" min="50" max="200" value={config.total_epochs}
                                onChange={(e) => setConfig({ ...config, total_epochs: parseInt(e.target.value) })}
                                className="w-full" />
                        </div>
                        <div>
                            <label className="text-[10px] tracking-widest block mb-1">NUM_STAGES: {config.num_stages}</label>
                            <input type="range" min="2" max="5" value={config.num_stages}
                                onChange={(e) => setConfig({ ...config, num_stages: parseInt(e.target.value) })}
                                className="w-full" />
                        </div>
                        <div>
                            <label className="text-[10px] tracking-widest block mb-1">STRATEGY</label>
                            <select value={config.strategy}
                                onChange={(e) => setConfig({ ...config, strategy: e.target.value })}
                                className="w-full bg-white/50 border-2 border-black px-3 py-2 text-sm">
                                <option value="linear">Linear</option>
                                <option value="exponential">Exponential</option>
                                <option value="step">Step</option>
                            </select>
                        </div>
                        <button onClick={build} disabled={loading} className="pixel-btn w-full">
                            {loading ? '...' : 'BUILD CURRICULUM'}
                        </button>
                    </div>
                </motion.div>

                <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="card">
                    <h2 className="text-sm font-bold tracking-widest mb-4">SCHEDULE</h2>
                    {result ? (
                        <div className="space-y-3">
                            {result.stages.map((stage, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: i * 0.1 }}
                                    className={`p-4 border-2 border-black ${getDifficultyColor(stage.difficulty)}`}
                                >
                                    <div className="flex justify-between mb-2">
                                        <span className="font-bold">Stage {i + 1}</span>
                                        <span className="label-tag text-[9px]">{stage.difficulty.toUpperCase()}</span>
                                    </div>
                                    <div className="text-[11px] space-y-1">
                                        <div>📅 Epochs {stage.epoch_start} → {stage.epoch_end}</div>
                                        <div>💪 Strength: {(stage.augmentation_strength * 100).toFixed(0)}%</div>
                                        <div className="flex flex-wrap gap-1 mt-2">
                                            {stage.transforms.map(t => (
                                                <span key={t} className="bg-black text-white px-2 py-0.5 text-[9px]">{t}</span>
                                            ))}
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-sm opacity-50 text-center py-8">Configure and build to see schedule</p>
                    )}
                </motion.div>
            </div>
        </div>
    )
}

export default Curriculum
