import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'

function Dashboard() {
    const [health, setHealth] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch('/api/health')
            .then(res => res.json())
            .then(data => {
                setHealth(data)
                setLoading(false)
            })
            .catch(() => setLoading(false))
    }, [])

    return (
        <div className="max-w-6xl mx-auto">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8"
            >
                <h1 className="text-2xl font-bold tracking-widest mb-2">
                    AUGMENTAI :: DASHBOARD
                </h1>
                <p className="text-sm opacity-70">
                    LLM-Powered Data Augmentation Policy Designer
                </p>
            </motion.div>

            {/* Status Bar */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="card mb-6 flex items-center gap-4"
            >
                <span className="text-[10px] tracking-widest">SYSTEM_STATUS:</span>
                {loading ? (
                    <span className="blink">⏳ CHECKING...</span>
                ) : health ? (
                    <>
                        <span className="text-green-600">✅ ONLINE</span>
                        <span className="text-[10px] opacity-70">
                            LLM: {health.llm_available ? '✅' : '❌'} |
                            Version: {health.version}
                        </span>
                    </>
                ) : (
                    <span className="text-red-600">❌ API OFFLINE</span>
                )}
            </motion.div>

            {/* Quick Actions Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                <QuickAction
                    to="/dataset"
                    icon="📁"
                    title="PREPARE DATASET"
                    desc="Upload and inspect datasets"
                    delay={0.3}
                />
                <QuickAction
                    to="/policy"
                    icon="🎨"
                    title="BUILD POLICY"
                    desc="Design augmentation pipelines"
                    delay={0.4}
                />
                <QuickAction
                    to="/search"
                    icon="🔍"
                    title="AUTO SEARCH"
                    desc="Find optimal policies"
                    delay={0.5}
                />
            </div>

            {/* Recent Projects (Placeholder) */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="card"
            >
                <h2 className="text-sm font-bold tracking-widest mb-4">
                    RECENT_PROJECTS
                </h2>
                <p className="text-sm opacity-50">
                    No projects yet. Start by preparing a dataset.
                </p>
            </motion.div>
        </div>
    )
}

function QuickAction({ to, icon, title, desc, delay }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay }}
        >
            <Link
                to={to}
                className="card block hover:translate-x-1 hover:-translate-y-1 
                   hover:shadow-[4px_4px_0_0_#000] transition-all"
            >
                <span className="text-2xl">{icon}</span>
                <h3 className="text-sm font-bold tracking-widest mt-2">{title}</h3>
                <p className="text-[10px] opacity-70 mt-1">{desc}</p>
            </Link>
        </motion.div>
    )
}

export default Dashboard
