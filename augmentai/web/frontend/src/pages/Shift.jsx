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
            <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
                <h1 className="text-2xl font-bold tracking-widest mb-2">DOMAIN :: SHIFT</h1>
                <p className="text-sm opacity-70">Simulate controlled distribution shifts</p>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="card">
                    <h2 className="text-sm font-bold tracking-widest mb-4">CONFIG</h2>
                    <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="text-[10px] tracking-widest block mb-1">SOURCE</label>
                                <input type="text" value={config.source_domain}
                                    onChange={(e) => setConfig({ ...config, source_domain: e.target.value })}
                                    className="w-full bg-white/50 border-2 border-black px-3 py-2 text-sm" />
                            </div>
                            <div>
                                <label className="text-[10px] tracking-widest block mb-1">TARGET</label>
                                <input type="text" value={config.target_domain}
                                    onChange={(e) => setConfig({ ...config, target_domain: e.target.value })}
                                    className="w-full bg-white/50 border-2 border-black px-3 py-2 text-sm" />
                            </div>
                        </div>
                        <div>
                            <label className="text-[10px] tracking-widest block mb-2">SHIFT_TYPES</label>
                            <div className="flex flex-wrap gap-2">
                                {shiftTypes.map(t => (
                                    <button
                                        key={t.id}
                                        onClick={() => toggleShiftType(t.id)}
                                        className={`px-3 py-1 text-[11px] border-2 border-black ${config.shift_types.includes(t.id) ? 'bg-black text-white' : 'bg-white/50'
                                            }`}
                                    >
                                        {t.name}
                                    </button>
                                ))}
                            </div>
                        </div>
                        <button onClick={simulate} disabled={loading} className="pixel-btn w-full">
                            {loading ? '...' : 'SIMULATE SHIFT'}
                        </button>
                    </div>
                </motion.div>

                <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="card">
                    <h2 className="text-sm font-bold tracking-widest mb-4">RESULTS</h2>
                    {result ? (
                        <div>
                            <div className="text-center mb-6">
                                <span className="text-[10px] tracking-widest block">ESTIMATED_PERF_DROP</span>
                                <span className="text-4xl font-bold text-red-600">
                                    -{(result.estimated_performance_drop * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div className="text-[10px] tracking-widest mb-2">SHIFTS_APPLIED</div>
                            <div className="space-y-2">
                                {result.shifts_applied.map((s, i) => (
                                    <div key={i} className="bg-black text-white p-3">
                                        <div className="flex justify-between">
                                            <span className="font-bold">{s.name}</span>
                                            <span className="opacity-70">{(s.severity * 100).toFixed(0)}%</span>
                                        </div>
                                        <p className="text-[10px] opacity-50 mt-1">{s.description}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <p className="text-sm opacity-50 text-center py-8">Configure and simulate to see results</p>
                    )}
                </motion.div>
            </div>
        </div>
    )
}

export default Shift
