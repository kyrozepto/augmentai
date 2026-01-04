import { Outlet, NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'

const navItems = [
    { path: '/', label: '🏠 DASHBOARD', exact: true },
    { path: '/dataset', label: '📁 DATASETS' },
    { path: '/policy', label: '🎨 POLICIES' },
    { path: '/search', label: '🔍 SEARCH' },
    { path: '/chat', label: '💬 CHAT' },
    { type: 'divider', label: 'ANALYSIS' },
    { path: '/ablation', label: '📊 ABLATION' },
    { path: '/diff', label: '🔀 DIFF' },
    { path: '/repair', label: '🔧 REPAIR' },
    { path: '/curriculum', label: '📈 CURRICULUM' },
    { path: '/shift', label: '🌐 SHIFT' },
]

function Layout() {
    return (
        <div className="flex min-h-screen">
            {/* Sidebar */}
            <nav className="sidebar flex flex-col">
                {/* Logo */}
                <div className="p-4 border-b border-white/20">
                    <motion.h1
                        className="text-lg font-bold tracking-widest"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        AUGMENT<span className="text-gray-400">AI</span>
                    </motion.h1>
                    <p className="text-[10px] text-gray-500 mt-1">WEB UI v1.1.0</p>
                </div>

                {/* Navigation Links */}
                <div className="flex-1 py-4">
                    {navItems.map((item, i) => (
                        item.type === 'divider' ? (
                            <div key={i} className="px-4 py-2 mt-2 text-[9px] text-gray-500 tracking-widest">
                                {item.label}
                            </div>
                        ) : (
                            <NavLink
                                key={item.path}
                                to={item.path}
                                end={item.exact}
                                className={({ isActive }) =>
                                    `sidebar-link ${isActive ? 'active bg-white/20' : ''}`
                                }
                            >
                                <motion.span
                                    initial={{ opacity: 0, x: -10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: i * 0.05 }}
                                >
                                    {item.label}
                                </motion.span>
                            </NavLink>
                        )
                    ))}
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-white/20 text-[10px] text-gray-500">
                    <p>⚙️ Settings</p>
                    <p className="mt-2">📖 Docs</p>
                </div>
            </nav>

            {/* Main Content */}
            <main className="flex-1 p-6 grid-bg">
                <Outlet />
            </main>
        </div>
    )
}

export default Layout
