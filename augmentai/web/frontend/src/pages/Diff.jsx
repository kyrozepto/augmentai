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

    const getTypeColor = (type) => {
        switch (type) {
            case 'added': return 'bg-green-200 border-green-600'
            case 'removed': return 'bg-red-200 border-red-600'
            case 'modified': return 'bg-yellow-200 border-yellow-600'
            default: return 'bg-gray-100 border-gray-400'
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
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8"
            >
                <h1 className="text-2xl font-bold tracking-widest mb-2">
                    POLICY :: DIFF
                </h1>
                <p className="text-sm opacity-70">
                    Compare two policies side-by-side
                </p>
            </motion.div>

            {/* Policy Inputs */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="card">
                    <h2 className="text-sm font-bold tracking-widest mb-2">POLICY_A (baseline)</h2>
                    <textarea
                        value={policyA}
                        onChange={(e) => setPolicyA(e.target.value)}
                        className="w-full h-40 bg-black text-white font-mono text-xs p-3"
                    />
                </motion.div>
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="card">
                    <h2 className="text-sm font-bold tracking-widest mb-2">POLICY_B (new)</h2>
                    <textarea
                        value={policyB}
                        onChange={(e) => setPolicyB(e.target.value)}
                        className="w-full h-40 bg-black text-white font-mono text-xs p-3"
                    />
                </motion.div>
            </div>

            <button onClick={runDiff} disabled={loading} className="pixel-btn w-full mb-6">
                {loading ? 'COMPARING...' : 'COMPARE POLICIES'}
            </button>

            {/* Results */}
            {result && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="card"
                >
                    {/* Summary */}
                    <div className="flex gap-4 mb-4">
                        <span className="label-tag bg-green-200">+{result.additions} added</span>
                        <span className="label-tag bg-red-200">-{result.removals} removed</span>
                        <span className="label-tag bg-yellow-200">~{result.modifications} modified</span>
                    </div>

                    {!result.has_changes ? (
                        <p className="text-green-600 text-center py-4">✓ No changes detected</p>
                    ) : (
                        <>
                            <p className="text-sm opacity-70 mb-4">{result.summary}</p>
                            <div className="space-y-2">
                                {result.entries.map((entry, i) => (
                                    <motion.div
                                        key={i}
                                        initial={{ opacity: 0, x: -10 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: i * 0.05 }}
                                        className={`border-l-4 p-3 ${getTypeColor(entry.type)}`}
                                    >
                                        <div className="flex items-center gap-2">
                                            <span className="font-mono text-lg w-6">{getTypeIcon(entry.type)}</span>
                                            <span className="font-bold">{entry.transform_name}</span>
                                        </div>
                                        {entry.type === 'modified' && (
                                            <div className="ml-8 text-[11px] mt-2 font-mono">
                                                <div className="text-red-600">- {JSON.stringify(entry.old_value)}</div>
                                                <div className="text-green-600">+ {JSON.stringify(entry.new_value)}</div>
                                            </div>
                                        )}
                                    </motion.div>
                                ))}
                            </div>
                        </>
                    )}
                </motion.div>
            )}
        </div>
    )
}

export default Diff
