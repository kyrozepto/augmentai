import { useState } from 'react'
import { motion } from 'framer-motion'

const SAMPLE_POLICY_A = `name: baseline_policy
domain: natural
transforms:
  - name: HorizontalFlip
    probability: 0.5
  - name: Rotate
    probability: 0.3
    parameters:
      limit: 15
  - name: GaussNoise
    probability: 0.2`

const SAMPLE_POLICY_B = `name: updated_policy
domain: natural
transforms:
  - name: HorizontalFlip
    probability: 0.5
  - name: Rotate
    probability: 0.5
    parameters:
      limit: 30
  - name: RandomBrightnessContrast
    probability: 0.4`

function Diff() {
    const [policyA, setPolicyA] = useState(SAMPLE_POLICY_A)
    const [policyB, setPolicyB] = useState(SAMPLE_POLICY_B)
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)

    const runDiff = async () => {
        setLoading(true)
        try {
            const res = await fetch('/api/diff', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    policy_a_yaml: policyA,
                    policy_b_yaml: policyB,
                }),
            })
            if (res.ok) {
                setResult(await res.json())
            }
        } catch (err) {
            console.error(err)
        }
        setLoading(false)
    }

    const getTypeStyles = (type) => {
        switch (type) {
            case 'added': return { border: 'border-green-400/50', bg: 'bg-green-400/10', text: 'text-green-400' }
            case 'removed': return { border: 'border-red-400/50', bg: 'bg-red-400/10', text: 'text-red-400' }
            case 'modified': return { border: 'border-yellow-400/50', bg: 'bg-yellow-400/10', text: 'text-yellow-400' }
            default: return { border: 'border-white/20', bg: 'bg-white/5', text: 'text-white/50' }
        }
    }

    const getTypeIcon = (type) => {
        switch (type) {
            case 'added': return '+'
            case 'removed': return '-'
            case 'modified': return '~'
            default: return '='
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
                        COMPARISON_MODULE
                    </span>
                </motion.div>
                <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white mb-2">
                    POLICY :: DIFF
                </h1>
                <p className="text-sm text-white/50">
                    Compare two policies side-by-side
                </p>
            </motion.div>

            {/* Policy Inputs */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
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
                        <span className="text-[10px] text-white/40">policy_a.yaml</span>
                    </div>
                    <div className="p-4">
                        <h2 className="text-[10px] font-bold tracking-wider text-white/50 mb-2">
                            POLICY_A (baseline)
                        </h2>
                        <textarea
                            value={policyA}
                            onChange={(e) => setPolicyA(e.target.value)}
                            className="w-full h-40 code-block resize-none text-xs"
                        />
                    </div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, x: 30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="card-window"
                >
                    <div className="card-header">
                        <div className="window-dots">
                            <div className="window-dot red" />
                            <div className="window-dot yellow" />
                            <div className="window-dot green" />
                        </div>
                        <span className="text-[10px] text-white/40">policy_b.yaml</span>
                    </div>
                    <div className="p-4">
                        <h2 className="text-[10px] font-bold tracking-wider text-white/50 mb-2">
                            POLICY_B (new)
                        </h2>
                        <textarea
                            value={policyB}
                            onChange={(e) => setPolicyB(e.target.value)}
                            className="w-full h-40 code-block resize-none text-xs"
                        />
                    </div>
                </motion.div>
            </div>

            <motion.button
                onClick={runDiff}
                disabled={loading}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                className="pixel-btn w-full mb-6"
            >
                {loading ? 'COMPARING...' : 'COMPARE_POLICIES →'}
            </motion.button>

            {/* Results */}
            {result && (
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="card"
                >
                    {/* Summary */}
                    <div className="flex gap-3 mb-6">
                        <span className="label-tag label-tag-success">+{result.additions} added</span>
                        <span className="label-tag label-tag-error">-{result.removals} removed</span>
                        <span className="label-tag label-tag-warning">~{result.modifications} modified</span>
                    </div>

                    {!result.has_changes ? (
                        <div className="text-center py-8">
                            <span className="text-green-400/80">✓ No changes detected</span>
                        </div>
                    ) : (
                        <>
                            <p className="text-sm text-white/50 mb-6">{result.summary}</p>
                            <div className="space-y-3">
                                {result.entries.map((entry, i) => {
                                    const styles = getTypeStyles(entry.type)
                                    return (
                                        <motion.div
                                            key={i}
                                            initial={{ opacity: 0, x: -20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: i * 0.05 }}
                                            className={`border-l-4 ${styles.border} ${styles.bg} p-4`}
                                        >
                                            <div className="flex items-center gap-3">
                                                <span className={`font-mono text-lg w-6 ${styles.text}`}>
                                                    {getTypeIcon(entry.type)}
                                                </span>
                                                <span className="font-bold text-white">{entry.transform_name}</span>
                                            </div>
                                            {entry.type === 'modified' && (
                                                <div className="ml-9 text-[11px] mt-3 font-mono space-y-1">
                                                    <div className="text-red-400/80">- {JSON.stringify(entry.old_value)}</div>
                                                    <div className="text-green-400/80">+ {JSON.stringify(entry.new_value)}</div>
                                                </div>
                                            )}
                                        </motion.div>
                                    )
                                })}
                            </div>
                        </>
                    )}
                </motion.div>
            )}
        </div>
    )
}

export default Diff
