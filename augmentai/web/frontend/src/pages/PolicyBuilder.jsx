import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

function PolicyBuilder() {
    const [domains, setDomains] = useState([])
    const [selectedDomain, setSelectedDomain] = useState('natural')
    const [transforms, setTransforms] = useState([])
    const [pipeline, setPipeline] = useState([])
    const [policyName, setPolicyName] = useState('my_policy')

    useEffect(() => {
        // Fetch available domains
        fetch('/api/domains')
            .then(res => res.json())
            .then(data => setDomains(data))
            .catch(console.error)
    }, [])

    useEffect(() => {
        // Fetch transforms for selected domain
        if (selectedDomain) {
            fetch(`/api/domains/${selectedDomain}/transforms`)
                .then(res => res.json())
                .then(data => setTransforms(data))
                .catch(console.error)
        }
    }, [selectedDomain])

    const addTransform = (transform) => {
        setPipeline([...pipeline, {
            ...transform,
            id: Date.now(),
            probability: 0.5,
        }])
    }

    const removeTransform = (id) => {
        setPipeline(pipeline.filter(t => t.id !== id))
    }

    const updateProbability = (id, prob) => {
        setPipeline(pipeline.map(t =>
            t.id === id ? { ...t, probability: prob } : t
        ))
    }

    const exportPolicy = async (format) => {
        const policy = {
            name: policyName,
            domain: selectedDomain,
            transforms: pipeline.map(t => ({
                name: t.name,
                probability: t.probability,
                parameters: {},
            })),
        }

        // Create policy on backend
        const res = await fetch('/api/policies', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(policy),
        })

        if (res.ok) {
            const data = await res.json()
            // Export it
            const exportRes = await fetch(`/api/policies/${data.id}/export?format=${format}`)
            const exportData = await exportRes.json()

            // Download as file
            const blob = new Blob([exportData.content], { type: 'text/plain' })
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `${policyName}.${format === 'python' ? 'py' : format}`
            a.click()
        }
    }

    return (
        <div className="max-w-6xl mx-auto">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8 flex justify-between items-start"
            >
                <div>
                    <h1 className="text-2xl font-bold tracking-widest mb-2">
                        POLICY :: BUILDER
                    </h1>
                    <p className="text-sm opacity-70">
                        Design domain-safe augmentation pipelines
                    </p>
                </div>
                <div className="flex gap-2">
                    <button onClick={() => exportPolicy('yaml')} className="pixel-btn pixel-btn-secondary text-[10px]">
                        EXPORT YAML
                    </button>
                    <button onClick={() => exportPolicy('python')} className="pixel-btn text-[10px]">
                        EXPORT PYTHON
                    </button>
                </div>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Domain & Transforms */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="card"
                >
                    <h2 className="text-sm font-bold tracking-widest mb-4">
                        DOMAIN
                    </h2>
                    <select
                        value={selectedDomain}
                        onChange={(e) => setSelectedDomain(e.target.value)}
                        className="w-full bg-white/50 border-2 border-black px-3 py-2 
                       text-sm font-mono mb-4"
                    >
                        {domains.map(d => (
                            <option key={d.id} value={d.id}>{d.name}</option>
                        ))}
                    </select>

                    <h2 className="text-sm font-bold tracking-widest mb-4">
                        AVAILABLE_TRANSFORMS
                    </h2>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                        {transforms.map(t => (
                            <button
                                key={t.name}
                                onClick={() => addTransform(t)}
                                className="w-full text-left px-3 py-2 border-2 border-black/30
                           bg-white/30 text-[11px] hover:border-black 
                           hover:bg-white/50 transition-colors"
                            >
                                {t.name}
                            </button>
                        ))}
                    </div>
                </motion.div>

                {/* Pipeline */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="card lg:col-span-2"
                >
                    <div className="flex justify-between items-center mb-4">
                        <h2 className="text-sm font-bold tracking-widest">
                            PIPELINE
                        </h2>
                        <input
                            type="text"
                            value={policyName}
                            onChange={(e) => setPolicyName(e.target.value)}
                            className="bg-transparent border-b-2 border-black px-2 py-1
                         text-sm font-mono focus:outline-none"
                            placeholder="policy_name"
                        />
                    </div>

                    {pipeline.length === 0 ? (
                        <p className="text-sm opacity-50 text-center py-8">
                            Click transforms on the left to add them to the pipeline
                        </p>
                    ) : (
                        <div className="space-y-3">
                            {pipeline.map((t, i) => (
                                <motion.div
                                    key={t.id}
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    className="flex items-center gap-4 p-3 border-2 border-black
                             bg-white/50"
                                >
                                    <span className="text-[10px] font-bold opacity-50">
                                        {i + 1}.
                                    </span>
                                    <div className="flex-1">
                                        <span className="text-sm font-bold">{t.name}</span>
                                        <div className="flex items-center gap-2 mt-1">
                                            <span className="text-[10px]">p=</span>
                                            <input
                                                type="range"
                                                min="0"
                                                max="1"
                                                step="0.1"
                                                value={t.probability}
                                                onChange={(e) => updateProbability(t.id, parseFloat(e.target.value))}
                                                className="flex-1"
                                            />
                                            <span className="text-[10px] w-8">{t.probability}</span>
                                        </div>
                                    </div>
                                    <button
                                        onClick={() => removeTransform(t.id)}
                                        className="text-red-600 hover:text-red-800"
                                    >
                                        🗑️
                                    </button>
                                </motion.div>
                            ))}
                        </div>
                    )}
                </motion.div>
            </div>
        </div>
    )
}

export default PolicyBuilder
