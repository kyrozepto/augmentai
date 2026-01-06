import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'

const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: { staggerChildren: 0.1, delayChildren: 0.2 }
    }
}

const fadeUp = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } }
}

const quickActions = [
    {
        to: '/dataset',
        icon: (
            <svg viewBox="0 0 24 24" className="w-full h-full">
                <rect x="2" y="2" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="1.5" />
                <rect x="13" y="2" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="1.5" />
                <rect x="2" y="13" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="1.5" />
                <rect x="13" y="13" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="1.5" />
                <circle cx="6.5" cy="6.5" r="2" fill="currentColor" />
            </svg>
        ),
        title: 'PREPARE_DATASET',
        desc: 'Upload and inspect datasets with intelligent analysis'
    },
    {
        to: '/policy',
        icon: (
            <svg viewBox="0 0 24 24" className="w-full h-full">
                <rect x="2" y="4" width="20" height="3" fill="currentColor" rx="1" />
                <rect x="2" y="10.5" width="14" height="3" fill="currentColor" rx="1" />
                <rect x="2" y="17" width="18" height="3" fill="currentColor" rx="1" />
                <circle cx="20" cy="12" r="2" fill="none" stroke="currentColor" strokeWidth="1.5" />
            </svg>
        ),
        title: 'BUILD_POLICY',
        desc: 'Design domain-safe augmentation pipelines'
    },
    {
        to: '/search',
        icon: (
            <svg viewBox="0 0 24 24" className="w-full h-full">
                <path d="M2 20 L6 14 L10 16 L14 8 L18 10 L22 4" stroke="currentColor" strokeWidth="2" fill="none" />
                <circle cx="6" cy="14" r="2" fill="currentColor" />
                <circle cx="10" cy="16" r="2" fill="currentColor" />
                <circle cx="14" cy="8" r="2" fill="currentColor" />
                <circle cx="18" cy="10" r="2" fill="currentColor" />
                <circle cx="22" cy="4" r="2" fill="currentColor" />
            </svg>
        ),
        title: 'AUTO_SEARCH',
        desc: 'Find optimal policies via evolutionary search'
    }
]

function Dashboard() {
    const [health, setHealth] = useState(null)
    const [loading, setLoading] = useState(true)
    const [hoveredAction, setHoveredAction] = useState(null)

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
                initial={{ opacity: 0, y: -30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="mb-10"
            >
                <motion.div
                    className="inline-block mb-4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                >
                    <span className="text-[10px] tracking-[0.2em] text-white/40 border border-white/20 px-3 py-1">
                        CONTROL_PANEL
                    </span>
                </motion.div>
                <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-white mb-3">
                    AUGMENTAI :: DASHBOARD
                </h1>
                <p className="text-sm text-white/50 max-w-lg">
                    LLM-Powered Data Augmentation Policy Designer. Configure, optimize, and export domain-safe augmentation pipelines.
                </p>
            </motion.div>

            {/* Status Bar */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="card-window mb-8"
            >
                <div className="card-header">
                    <div className="window-dots">
                        <div className="window-dot red" />
                        <div className="window-dot yellow" />
                        <div className="window-dot green" />
                    </div>
                    <span className="text-[10px] text-white/40">system_status.log</span>
                </div>
                <div className="p-4 flex items-center gap-6">
                    <span className="text-[10px] tracking-[0.15em] text-white/40">SYSTEM_STATUS:</span>
                    {loading ? (
                        <span className="text-white/60 text-sm blink">⏳ CHECKING...</span>
                    ) : health ? (
                        <div className="flex items-center gap-6">
                            <span className="flex items-center gap-2 text-sm">
                                <span className="status-dot online" />
                                <span className="text-green-400/80">ONLINE</span>
                            </span>
                            <span className="text-[11px] text-white/40">
                                LLM: {health.llm_available ? (
                                    <span className="text-green-400/70">AVAILABLE</span>
                                ) : (
                                    <span className="text-red-400/70">UNAVAILABLE</span>
                                )}
                            </span>
                            <span className="text-[11px] text-white/30">
                                VERSION: {health.version}
                            </span>
                        </div>
                    ) : (
                        <span className="flex items-center gap-2 text-sm">
                            <span className="status-dot offline" />
                            <span className="text-red-400/80">API OFFLINE</span>
                        </span>
                    )}
                </div>
            </motion.div>

            {/* Quick Actions Grid */}
            <motion.div
                className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10"
                initial="hidden"
                animate="visible"
                variants={staggerContainer}
            >
                {quickActions.map((action, i) => (
                    <motion.div
                        key={action.to}
                        variants={fadeUp}
                        whileHover={{ y: -8, borderColor: 'rgba(255,255,255,0.5)' }}
                        onHoverStart={() => setHoveredAction(action.to)}
                        onHoverEnd={() => setHoveredAction(null)}
                    >
                        <Link
                            to={action.to}
                            className="feature-card block h-full"
                        >
                            <motion.div
                                className="feature-icon"
                                animate={hoveredAction === action.to ? { rotate: [0, -5, 5, 0] } : {}}
                                transition={{ duration: 0.4 }}
                            >
                                {action.icon}
                            </motion.div>
                            <h3 className="text-sm font-bold text-white mb-2 tracking-wide">
                                {action.title}
                            </h3>
                            <p className="text-xs text-white/50 leading-relaxed">
                                {action.desc}
                            </p>
                            <motion.div
                                className="mt-4 text-xs text-white/30"
                                animate={{ opacity: hoveredAction === action.to ? 1 : 0 }}
                            >
                                LAUNCH →
                            </motion.div>
                        </Link>
                    </motion.div>
                ))}
            </motion.div>

            {/* Recent Projects */}
            <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="card"
            >
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-sm font-bold tracking-[0.15em] text-white">
                        RECENT_PROJECTS
                    </h2>
                    <span className="text-[10px] text-white/30 border border-white/20 px-2 py-1">
                        0 ITEMS
                    </span>
                </div>

                <div className="text-center py-12 border border-dashed border-white/10">
                    <motion.div
                        className="w-12 h-12 mx-auto mb-4 text-white/20"
                        animate={{ rotate: [0, 5, -5, 0] }}
                        transition={{ duration: 4, repeat: Infinity }}
                    >
                        <svg viewBox="0 0 24 24" className="w-full h-full">
                            <rect x="3" y="3" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3 2" />
                            <path d="M3 9 L21 9" stroke="currentColor" strokeWidth="1.5" />
                            <circle cx="6" cy="6" r="1" fill="currentColor" />
                        </svg>
                    </motion.div>
                    <p className="text-sm text-white/40 mb-2">
                        No projects yet
                    </p>
                    <p className="text-xs text-white/25">
                        Start by preparing a dataset or building a policy
                    </p>
                </div>
            </motion.div>

            {/* Decorative Elements */}
            <motion.div
                className="decorative-box w-20 h-20 -top-4 -right-4"
                initial={{ opacity: 0, rotate: 0 }}
                animate={{ opacity: 0.3, rotate: 12 }}
                transition={{ delay: 0.8 }}
                style={{ position: 'fixed', top: '10%', right: '5%' }}
            />
            <motion.div
                className="decorative-box w-12 h-12"
                initial={{ opacity: 0, rotate: 0 }}
                animate={{ opacity: 0.2, rotate: -8 }}
                transition={{ delay: 1 }}
                style={{ position: 'fixed', bottom: '15%', right: '10%' }}
            />
        </div>
    )
}

export default Dashboard
