import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

function Datasets() {
    const [datasets, setDatasets] = useState([])
    const [loading, setLoading] = useState(true)
    const [path, setPath] = useState('')
    const [registering, setRegistering] = useState(false)

    useEffect(() => {
        fetchDatasets()
    }, [])

    const fetchDatasets = () => {
        fetch('/api/datasets')
            .then(res => res.json())
            .then(data => {
                setDatasets(data)
                setLoading(false)
            })
            .catch(() => setLoading(false))
    }

    const handleRegister = async (e) => {
        e.preventDefault()
        if (!path) return

        setRegistering(true)
        try {
            const formData = new FormData()
            formData.append('path', path)

            const res = await fetch('/api/datasets', {
                method: 'POST',
                body: formData,
            })

            if (res.ok) {
                setPath('')
                fetchDatasets()
            }
        } catch (err) {
            console.error(err)
        }
        setRegistering(false)
    }

    return (
        <div className="max-w-6xl mx-auto">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8"
            >
                <h1 className="text-2xl font-bold tracking-widest mb-2">
                    DATASETS :: BROWSER
                </h1>
                <p className="text-sm opacity-70">
                    Upload, inspect, and manage your datasets
                </p>
            </motion.div>

            {/* Register Dataset Form */}
            <motion.form
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                onSubmit={handleRegister}
                className="card mb-6"
            >
                <h2 className="text-sm font-bold tracking-widest mb-4">
                    REGISTER_DATASET
                </h2>
                <div className="flex gap-4">
                    <input
                        type="text"
                        value={path}
                        onChange={(e) => setPath(e.target.value)}
                        placeholder="Enter dataset path (e.g., ./dataset)"
                        className="flex-1 bg-white/50 border-2 border-black px-4 py-2 
                       text-sm font-mono focus:outline-none focus:bg-white"
                    />
                    <button
                        type="submit"
                        disabled={registering}
                        className="pixel-btn"
                    >
                        {registering ? 'SCANNING...' : 'REGISTER'}
                    </button>
                </div>
            </motion.form>

            {/* Datasets List */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="card"
            >
                <h2 className="text-sm font-bold tracking-widest mb-4">
                    REGISTERED_DATASETS
                </h2>

                {loading ? (
                    <p className="text-sm blink">⏳ LOADING...</p>
                ) : datasets.length === 0 ? (
                    <p className="text-sm opacity-50">
                        No datasets registered. Enter a path above to add one.
                    </p>
                ) : (
                    <div className="space-y-4">
                        {datasets.map((ds, i) => (
                            <motion.div
                                key={ds.id}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.1 * i }}
                                className="border-2 border-black/30 p-4 bg-white/30
                           hover:border-black transition-colors"
                            >
                                <div className="flex justify-between items-start">
                                    <div>
                                        <h3 className="text-sm font-bold tracking-widest">
                                            {ds.name}
                                        </h3>
                                        <p className="text-[10px] opacity-70 mt-1">{ds.path}</p>
                                    </div>
                                    <span className="label-tag">
                                        {ds.domain?.toUpperCase() || 'NATURAL'}
                                    </span>
                                </div>
                                <div className="flex gap-4 mt-3 text-[10px]">
                                    <span>📷 {ds.image_count} images</span>
                                    {ds.classes.length > 0 && (
                                        <span>🏷️ {ds.classes.length} classes</span>
                                    )}
                                </div>
                            </motion.div>
                        ))}
                    </div>
                )}
            </motion.div>
        </div>
    )
}

export default Datasets
