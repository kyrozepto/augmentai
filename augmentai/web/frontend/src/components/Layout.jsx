import { Outlet, NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useState } from 'react'

const navItems = [
    { path: '/', label: 'DASHBOARD', icon: '◊', exact: true },
    { path: '/dataset', label: 'DATASETS', icon: '◫' },
    { path: '/policy', label: 'POLICIES', icon: '◈' },
    { path: '/search', label: 'SEARCH', icon: '◎' },
    { path: '/chat', label: 'CHAT', icon: '◇' },
    { type: 'divider', label: '// ANALYSIS' },
    { path: '/ablation', label: 'ABLATION', icon: '◆' },
    { path: '/diff', label: 'DIFF', icon: '◁' },
    { path: '/repair', label: 'REPAIR', icon: '◐' },
    { path: '/curriculum', label: 'CURRICULUM', icon: '◑' },
    { path: '/shift', label: 'SHIFT', icon: '◒' },
]

const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: {
            staggerChildren: 0.05,
            delayChildren: 0.1
        }
    }
}

const fadeInItem = {
    hidden: { opacity: 0, x: -20 },
    visible: {
        opacity: 1,
        x: 0,
        transition: { duration: 0.3, ease: 'easeOut' }
    }
}

function Layout() {
    const [hoveredItem, setHoveredItem] = useState(null)

    return (
        <div className="flex min-h-screen bg-black">
            {/* Sidebar */}
            <nav className="sidebar flex flex-col">
                {/* Logo */}
                <motion.div
                    className="p-5 border-b border-white/10"
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <div className="flex items-center gap-3 mb-2">
                        <motion.div
                            className="w-2 h-2 bg-green-500"
                            animate={{ opacity: [1, 0.5, 1] }}
                            transition={{ duration: 2, repeat: Infinity }}
                        />
                        <h1 className="text-base font-bold tracking-[0.2em] text-white">
                            AUGMENT<span className="text-white/40">AI</span>
                        </h1>
                    </div>
                    <p className="text-[10px] text-white/30 tracking-wider">
                        WEB_UI :: v1.1.0
                    </p>
                </motion.div>

                {/* Navigation Links */}
                <motion.div
                    className="flex-1 py-4"
                    initial="hidden"
                    animate="visible"
                    variants={staggerContainer}
                >
                    {navItems.map((item, i) => (
                        item.type === 'divider' ? (
                            <motion.div
                                key={i}
                                variants={fadeInItem}
                                className="px-5 py-3 mt-4 text-[9px] text-white/30 tracking-[0.15em]"
                            >
                                {item.label}
                            </motion.div>
                        ) : (
                            <motion.div key={item.path} variants={fadeInItem}>
                                <NavLink
                                    to={item.path}
                                    end={item.exact}
                                    className={({ isActive }) =>
                                        `sidebar-link flex items-center gap-3 ${isActive ? 'active' : ''}`
                                    }
                                    onMouseEnter={() => setHoveredItem(item.path)}
                                    onMouseLeave={() => setHoveredItem(null)}
                                >
                                    <motion.span
                                        className="text-xs opacity-50"
                                        animate={{
                                            rotate: hoveredItem === item.path ? [0, -10, 10, 0] : 0
                                        }}
                                        transition={{ duration: 0.4 }}
                                    >
                                        {item.icon}
                                    </motion.span>
                                    <span>{item.label}</span>
                                    {hoveredItem === item.path && (
                                        <motion.span
                                            className="ml-auto text-white/30"
                                            initial={{ opacity: 0, x: -5 }}
                                            animate={{ opacity: 1, x: 0 }}
                                        >
                                            →
                                        </motion.span>
                                    )}
                                </NavLink>
                            </motion.div>
                        )
                    ))}
                </motion.div>

                {/* Footer */}
                <motion.div
                    className="p-5 border-t border-white/10"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                >
                    <div className="space-y-2">
                        <motion.a
                            href="#"
                            className="flex items-center gap-2 text-[10px] text-white/40 hover:text-white/70 transition-colors"
                            whileHover={{ x: 4 }}
                        >
                            <span>◪</span>
                            <span>SETTINGS</span>
                        </motion.a>
                        <motion.a
                            href="https://github.com"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 text-[10px] text-white/40 hover:text-white/70 transition-colors"
                            whileHover={{ x: 4 }}
                        >
                            <span>◧</span>
                            <span>DOCS</span>
                        </motion.a>
                    </div>
                    <div className="mt-4 pt-3 border-t border-white/5">
                        <p className="text-[9px] text-white/20">
                            © 2024 AugmentAI
                        </p>
                    </div>
                </motion.div>
            </nav>

            {/* Main Content */}
            <main className="flex-1 p-8 grid-bg overflow-auto">
                <Outlet />
            </main>
        </div>
    )
}

export default Layout
