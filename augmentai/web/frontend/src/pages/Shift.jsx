import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

function Shift() {
    const [shiftTypes, setShiftTypes] = useState([])
    const [config, setConfig] = useState({
        source_domain: 'training',
        target_domain: 'production',
        shift_types: ['brightness', 'contrast', 'noise'],
    })
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        fetch('/api/shift/types')
            .then(res => res.json())
            .then(data => setShiftTypes(data))
            .catch(console.error)
    }, [])

    const simulate = async () => {
        setLoading(true)
        try {
            const res = await fetch('/api/shift', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            })
            if (res.ok) setResult(await res.json())
        } catch (e) { console.error(e) }
        setLoading(false)
    }

    const toggleShiftType = (id) => {
        if (config.shift_types.includes(id)) {
            setConfig({ ...config, shift_types: config.shift_types.filter(t => t !== id) })
        } else {
            setConfig({ ...config, shift_types: [...config.shift_types, id] })
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
                        SHIFT_MODULE
                    </span>
                </motion.div>
                <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white mb-2">
                    DOMAIN :: SHIFT
                </h1>
                <p className="text-sm text-white/50">
                    Simulate controlled distribution shifts between domains
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
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="text-[10px] tracking-wider text-white/40 block mb-2">
                                    SOURCE
                                </label>
                                <input
                                    type="text"
                                    value={config.source_domain}
                                    onChange={(e) => setConfig({ ...config, source_domain: e.target.value })}
                                    className="w-full"
                                />
                            </div>
                            <div>
                                <label className="text-[10px] tracking-wider text-white/40 block mb-2">
                                    TARGET
                                </label>
                                <input
                                    type="text"
                                    value={config.target_domain}
                                    onChange={(e) => setConfig({ ...config, target_domain: e.target.value })}
                                    className="w-full"
                                />
                            </div>
                        </div>

                        <div>
                            <label className="text-[10px] tracking-wider text-white/40 block mb-3">
                                SHIFT_TYPES
                            </label>
                            <div className="flex flex-wrap gap-2">
                                {shiftTypes.map(t => (
                                    <motion.button
                                        key={t.id}
                                        onClick={() => toggleShiftType(t.id)}
                                        whileHover={{ scale: 1.05 }}
                                        whileTap={{ scale: 0.95 }}
                                        className={`px-3 py-1.5 text-[11px] border transition-all ${config.shift_types.includes(t.id)
                                                ? 'bg-white text-black border-white'
                                                : 'bg-transparent text-white/50 border-white/20 hover:border-white/50'
                                            }`}
                                    >
                                        {t.name}
                                    </motion.button>
                                ))}
                            </div>
                        </div>

                        <motion.button
                            onClick={simulate}
                            disabled={loading}
                            whileHover={{ scale: 1.02, y: -2 }}
                            whileTap={{ scale: 0.98 }}
                            className="pixel-btn w-full"
                        >
                            {loading ? 'SIMULATING...' : 'SIMULATE_SHIFT →'}
                        </motion.button>
                    </div>
                </motion.div>

                {/* Results */}
                <motion.div
                    initial={{ opacity: 0, x: 30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="card"
                >
                    <h2 className="text-sm font-bold tracking-[0.12em] text-white mb-6">
                        RESULTS
                    </h2>

                    {result ? (
                        <div>
                            <div className="text-center mb-8 py-6 border border-red-400/30 bg-red-400/10">
                                <span className="text-[10px] tracking-wider text-white/40 block mb-2">
                                    ESTIMATED_PERF_DROP
                                </span>
                                <motion.span
                                    className="text-5xl font-bold text-red-400/90"
                                    initial={{ scale: 1.2 }}
                                    animate={{ scale: 1 }}
                                >
                                    -{(result.estimated_performance_drop * 100).toFixed(1)}%
                                </motion.span>
                            </div>

                            <div className="text-[10px] tracking-wider text-white/40 mb-3">
                                SHIFTS_APPLIED
                            </div>
                            <div className="space-y-2">
                                {result.shifts_applied.map((s, i) => (
                                    <motion.div
                                        key={i}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: i * 0.08 }}
                                        whileHover={{ x: 4, borderColor: 'rgba(255,255,255,0.3)' }}
                                        className="p-4 border border-white/15 bg-white/5 transition-colors"
                                    >
                                        <div className="flex justify-between mb-2">
                                            <span className="font-bold text-white">{s.name}</span>
                                            <span className="text-white/50 text-sm">
                                                {(s.severity * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                        <p className="text-[10px] text-white/40">{s.description}</p>
                                    </motion.div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="text-center py-12">
                            <motion.div
                                className="w-16 h-16 mx-auto mb-4 text-white/15"
                                animate={{ rotate: [0, 10, -10, 0] }}
                                transition={{ duration: 4, repeat: Infinity }}
                            >
                                <svg viewBox="0 0 24 24" className="w-full h-full">
                                    <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="1" />
                                    <path d="M12 3 L12 21" stroke="currentColor" strokeWidth="1" />
                                    <path d="M3 12 L21 12" stroke="currentColor" strokeWidth="1" />
                                    <ellipse cx="12" cy="12" rx="4" ry="9" fill="none" stroke="currentColor" strokeWidth="1" />
                                </svg>
                            </motion.div>
                            <p className="text-sm text-white/30">
                                Configure and simulate to see results
                            </p>
                        </div>
                    )}
                </motion.div>
            </div>
        </div>
    )
}

export default Shift
