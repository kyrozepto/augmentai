import { useState } from 'react'
import { motion } from 'framer-motion'

const SAMPLE_POLICY = `name: sample_policy
domain: natural
transforms:
  - name: HorizontalFlip
    probability: 0.5
  - name: Rotate
    probability: 0.3
    parameters:
      limit: 15
  - name: RandomBrightnessContrast
    probability: 0.4
  - name: GaussNoise
    probability: 0.2`

function Ablation() {
    const [policyYaml, setPolicyYaml] = useState(SAMPLE_POLICY)
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)

    const runAblation = async () => {
        setLoading(true)
        try {
            const res = await fetch('/api/ablation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ policy_yaml: policyYaml, mock: true }),
            })
            if (res.ok) {
                setResult(await res.json())
            }
        } catch (err) {
            console.error(err)
        }
        setLoading(false)
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
                        ANALYSIS_MODULE
                    </span>
                </motion.div>
                <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white mb-2">
                    ABLATION :: ANALYZER
                </h1>
                <p className="text-sm text-white/50">
                    Measure transform contributions via leave-one-out analysis
                </p>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Policy Input */}
                <motion.div
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="card-window"
                >
                    <div className="card-header">
                        <div className="window-dots">
                            <div className="window-dot red" />
                            <div className="window-dot yellow" />
                            <div className="window-dot green" />
                        </div>
                        <span className="text-[10px] text-white/40">policy_input.yaml</span>
                    </div>
                    <div className="p-5">
                        <h2 className="text-sm font-bold tracking-[0.12em] text-white mb-4">
                            POLICY_YAML
                        </h2>
                        <textarea
                            value={policyYaml}
                            onChange={(e) => setPolicyYaml(e.target.value)}
                            className="w-full h-64 code-block resize-none"
                            placeholder="Paste policy YAML here..."
                        />
                        <motion.button
                            onClick={runAblation}
                            disabled={loading || !policyYaml.trim()}
                            whileHover={{ scale: 1.02, y: -2 }}
                            whileTap={{ scale: 0.98 }}
                            className="pixel-btn w-full mt-4"
                        >
                            {loading ? 'ANALYZING...' : 'RUN_ABLATION →'}
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
                        CONTRIBUTIONS
                    </h2>

                    {result ? (
                        <div className="space-y-4">
                            <div className="text-center py-4 border border-white/10 bg-white/5">
                                <span className="text-[10px] tracking-wider text-white/40">BASELINE_SCORE</span>
                                <p className="text-4xl font-bold text-white mt-2">
                                    {(result.baseline_score * 100).toFixed(1)}%
                                </p>
                            </div>

                            <div className="space-y-2">
                                {result.contributions.map((c, i) => (
                                    <motion.div
                                        key={c.name}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: i * 0.08 }}
                                        className="border border-white/15 p-4 bg-white/5"
                                    >
                                        <div className="flex justify-between mb-2">
                                            <span className="font-bold text-sm text-white">
                                                <span className="text-white/30">#{c.rank}</span> {c.name}
                                            </span>
                                            <span className={`text-sm font-bold ${c.is_helpful ? 'text-green-400/80' : 'text-red-400/80'}`}>
                                                {c.contribution >= 0 ? '+' : ''}{(c.contribution * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <div className="flex justify-between text-[10px] text-white/40">
                                            <span>Impact: {c.impact_label}</span>
                                            <span>Ablated: {(c.ablated_score * 100).toFixed(1)}%</span>
                                        </div>
                                    </motion.div>
                                ))}
                            </div>

                            <div className="text-[10px] mt-4 pt-4 border-t border-white/10 space-y-2">
                                {result.recommended_keeps?.length > 0 && (
                                    <p className="text-green-400/70">
                                        ✓ Most helpful: {result.recommended_keeps.join(', ')}
                                    </p>
                                )}
                                {result.recommended_removes?.length > 0 && (
                                    <p className="text-yellow-400/70">
                                        ⚠ Consider removing: {result.recommended_removes.join(', ')}
                                    </p>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div className="text-center py-12">
                            <motion.div
                                className="w-16 h-16 mx-auto mb-4 text-white/15"
                                animate={{ rotate: [0, 5, -5, 0] }}
                                transition={{ duration: 4, repeat: Infinity }}
                            >
                                <svg viewBox="0 0 24 24" className="w-full h-full">
                                    <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="1" />
                                    <path d="M12 6 L12 12 L16 14" stroke="currentColor" strokeWidth="1.5" fill="none" />
                                </svg>
                            </motion.div>
                            <p className="text-sm text-white/30">
                                Paste a policy and run ablation
                            </p>
                        </div>
                    )}
                </motion.div>
            </div>
        </div>
    )
}

export default Ablation
