import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: { staggerChildren: 0.08, delayChildren: 0.2 }
    }
}

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } }
}

function Datasets() {
    const [datasets, setDatasets] = useState([])
    const [loading, setLoading] = useState(true)
    const [path, setPath] = useState('')
    const [registering, setRegistering] = useState(false)
    const [selectedDataset, setSelectedDataset] = useState(null)
    const [datasetDetails, setDatasetDetails] = useState(null)
    const [thumbnails, setThumbnails] = useState([])
    const [distribution, setDistribution] = useState([])
    const [loadingDetails, setLoadingDetails] = useState(false)
    const folderInputRef = useRef(null)

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

    // State for selected folder feedback
    const [selectedFolderName, setSelectedFolderName] = useState('')

    // Handle folder selection from file picker
    const handleFolderSelect = (event) => {
        const files = event.target.files
        if (files && files.length > 0) {
            // Get the folder path from the first file's webkitRelativePath
            const firstFile = files[0]
            const relativePath = firstFile.webkitRelativePath
            const folderName = relativePath.split('/')[0]

            // Show the selected folder name
            setSelectedFolderName(folderName)

            // Pre-fill with a placeholder path that user should update
            setPath(`C:/path/to/${folderName}`)
        }
    }

    // Handle browse button click - uses File System Access API if available
    const handleBrowseClick = async () => {
        // Try modern File System Access API first (Chrome 86+, Edge 86+)
        if ('showDirectoryPicker' in window) {
            try {
                const dirHandle = await window.showDirectoryPicker({
                    mode: 'read',
                })
                // Get the directory name
                const folderName = dirHandle.name
                setSelectedFolderName(folderName)

                // Pre-fill with placeholder - user needs to enter full path
                // Browser security doesn't allow access to full filesystem path
                setPath(`C:/path/to/${folderName}`)
                return
            } catch (err) {
                // User cancelled or API not supported, fall back to input
                if (err.name !== 'AbortError') {
                    console.log('File System Access API not supported, using fallback')
                }
                return
            }
        }

        // Fallback to traditional folder input
        if (folderInputRef.current) {
            folderInputRef.current.click()
        }
    }

    const selectDataset = async (dataset) => {
        setSelectedDataset(dataset.id)
        setLoadingDetails(true)
        setThumbnails([])
        setDistribution([])

        try {
            // Fetch detailed stats
            const detailsRes = await fetch(`/api/datasets/${dataset.id}`)
            if (detailsRes.ok) {
                setDatasetDetails(await detailsRes.json())
            }

            // Fetch sample thumbnails
            const thumbRes = await fetch(`/api/datasets/${dataset.id}/sample-thumbnails?count=6`)
            if (thumbRes.ok) {
                setThumbnails(await thumbRes.json())
            }

            // Fetch class distribution
            const distRes = await fetch(`/api/datasets/${dataset.id}/distribution`)
            if (distRes.ok) {
                setDistribution(await distRes.json())
            }
        } catch (err) {
            console.error(err)
        }
        setLoadingDetails(false)
    }

    const formatBytes = (bytes) => {
        if (bytes === 0) return '0 B'
        const k = 1024
        const sizes = ['B', 'KB', 'MB', 'GB']
        const i = Math.floor(Math.log(bytes) / Math.log(k))
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    }

    const maxCount = distribution.length > 0 ? Math.max(...distribution.map(d => d.count)) : 1

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
                        DATA_MODULE
                    </span>
                </motion.div>
                <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white mb-2">
                    DATASETS :: BROWSER
                </h1>
                <p className="text-sm text-white/50">
                    Upload, inspect, and manage your datasets with intelligent analysis
                </p>
            </motion.div>

            {/* Register Dataset Form */}
            <motion.form
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                onSubmit={handleRegister}
                className="card-window mb-8"
            >
                <div className="card-header">
                    <div className="window-dots">
                        <div className="window-dot red" />
                        <div className="window-dot yellow" />
                        <div className="window-dot green" />
                    </div>
                    <span className="text-[10px] text-white/40">register_dataset.sh</span>
                </div>
                <div className="p-5">
                    <h2 className="text-sm font-bold tracking-[0.12em] text-white mb-4">
                        REGISTER_DATASET
                    </h2>

                    {/* Hidden folder input for fallback */}
                    <input
                        ref={folderInputRef}
                        type="file"
                        webkitdirectory=""
                        directory=""
                        multiple
                        onChange={handleFolderSelect}
                        className="hidden"
                    />

                    <div className="flex gap-3">
                        <div className="flex-1 flex gap-2">
                            <input
                                type="text"
                                value={path}
                                onChange={(e) => setPath(e.target.value)}
                                placeholder="Enter dataset path or click BROWSE"
                                className="flex-1"
                            />
                            <motion.button
                                type="button"
                                onClick={handleBrowseClick}
                                whileHover={{ scale: 1.02, y: -2 }}
                                whileTap={{ scale: 0.98 }}
                                className="pixel-btn pixel-btn-secondary pixel-btn-sm flex items-center gap-2"
                            >
                                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                                </svg>
                                BROWSE
                            </motion.button>
                        </div>
                        <motion.button
                            type="submit"
                            disabled={registering || !path}
                            whileHover={{ scale: 1.02, y: -2 }}
                            whileTap={{ scale: 0.98 }}
                            className="pixel-btn"
                        >
                            {registering ? 'SCANNING...' : 'REGISTER →'}
                        </motion.button>
                    </div>

                    {/* Selected folder feedback */}
                    {selectedFolderName ? (
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="mt-4 p-3 border border-yellow-400/30 bg-yellow-400/10"
                        >
                            <div className="flex items-center gap-2 mb-2">
                                <span className="text-yellow-400">📁</span>
                                <span className="text-sm font-bold text-yellow-400/80">
                                    Selected: {selectedFolderName}
                                </span>
                            </div>
                            <p className="text-[11px] text-white/60">
                                ⚠️ Browser security prevents access to full paths.
                                Please update the path above to the <strong className="text-white/80">full folder path</strong> on your system, e.g.:
                            </p>
                            <code className="block mt-2 text-[11px] text-green-400/80 bg-black/50 p-2">
                                C:\Users\you\Documents\{selectedFolderName}
                            </code>
                        </motion.div>
                    ) : (
                        <p className="text-[10px] text-white/30 mt-3">
                            💡 Enter the full path to your dataset folder (e.g., C:\Users\you\dataset)
                        </p>
                    )}
                </div>
            </motion.form>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Datasets List */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="card"
                >
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-sm font-bold tracking-[0.12em] text-white">
                            REGISTERED_DATASETS
                        </h2>
                        <span className="text-[10px] text-white/30 border border-white/20 px-2 py-1">
                            {datasets.length} ITEMS
                        </span>
                    </div>

                    {loading ? (
                        <div className="text-center py-8">
                            <span className="text-white/50 blink">⏳ LOADING...</span>
                        </div>
                    ) : datasets.length === 0 ? (
                        <div className="text-center py-12 border border-dashed border-white/10">
                            <motion.div
                                className="w-12 h-12 mx-auto mb-4 text-white/15"
                                animate={{ scale: [1, 1.05, 1] }}
                                transition={{ duration: 2, repeat: Infinity }}
                            >
                                <svg viewBox="0 0 24 24" className="w-full h-full">
                                    <rect x="2" y="2" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="1.5" />
                                    <rect x="13" y="2" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="1.5" />
                                    <rect x="2" y="13" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="1.5" />
                                    <rect x="13" y="13" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="1.5" />
                                </svg>
                            </motion.div>
                            <p className="text-sm text-white/30">
                                No datasets registered
                            </p>
                            <p className="text-xs text-white/20 mt-1">
                                Enter a path above to add one
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-2 max-h-[400px] overflow-y-auto">
                            {datasets.map((ds, i) => (
                                <motion.div
                                    key={ds.id}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: i * 0.05 }}
                                    whileHover={{ x: 4 }}
                                    onClick={() => selectDataset(ds)}
                                    className={`border p-4 cursor-pointer transition-all ${selectedDataset === ds.id
                                        ? 'border-white/50 bg-white/10'
                                        : 'border-white/15 bg-white/5 hover:border-white/30'
                                        }`}
                                >
                                    <div className="flex justify-between items-start">
                                        <div>
                                            <h3 className="text-sm font-bold tracking-wide text-white">
                                                {ds.name}
                                            </h3>
                                            <p className="text-[10px] text-white/40 mt-1 font-mono truncate max-w-[200px]">
                                                {ds.path}
                                            </p>
                                        </div>
                                        <span className="label-tag text-[9px]">
                                            {ds.domain?.toUpperCase() || 'NATURAL'}
                                        </span>
                                    </div>
                                    <div className="flex gap-4 mt-3 text-[10px] text-white/50">
                                        <span>◫ {ds.image_count} images</span>
                                        {ds.classes?.length > 0 && (
                                            <span>◈ {ds.classes.length} classes</span>
                                        )}
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    )}
                </motion.div>

                {/* Dataset Details Panel */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="card"
                >
                    <h2 className="text-sm font-bold tracking-[0.12em] text-white mb-6">
                        DATASET_DETAILS
                    </h2>

                    {!selectedDataset ? (
                        <div className="text-center py-12">
                            <motion.div
                                className="w-16 h-16 mx-auto mb-4 text-white/15"
                                animate={{ rotate: [0, 5, -5, 0] }}
                                transition={{ duration: 4, repeat: Infinity }}
                            >
                                <svg viewBox="0 0 24 24" className="w-full h-full">
                                    <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="1" />
                                    <path d="M12 8 L12 12 L16 12" stroke="currentColor" strokeWidth="1.5" fill="none" />
                                </svg>
                            </motion.div>
                            <p className="text-sm text-white/30">
                                Select a dataset to view details
                            </p>
                        </div>
                    ) : loadingDetails ? (
                        <div className="text-center py-12">
                            <span className="text-white/50 blink">⏳ LOADING DETAILS...</span>
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {/* Stats Grid */}
                            {datasetDetails && (
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="p-3 border border-white/10 bg-white/5">
                                        <span className="text-[9px] text-white/40 tracking-wider">IMAGES</span>
                                        <p className="text-xl font-bold text-white">{datasetDetails.image_count}</p>
                                    </div>
                                    <div className="p-3 border border-white/10 bg-white/5">
                                        <span className="text-[9px] text-white/40 tracking-wider">SIZE</span>
                                        <p className="text-xl font-bold text-white">{formatBytes(datasetDetails.total_size_bytes)}</p>
                                    </div>
                                    <div className="p-3 border border-white/10 bg-white/5">
                                        <span className="text-[9px] text-white/40 tracking-wider">RESOLUTION</span>
                                        <p className="text-sm font-bold text-white">{datasetDetails.avg_resolution}</p>
                                    </div>
                                    <div className="p-3 border border-white/10 bg-white/5">
                                        <span className="text-[9px] text-white/40 tracking-wider">CLASSES</span>
                                        <p className="text-xl font-bold text-white">{datasetDetails.classes?.length || 0}</p>
                                    </div>
                                </div>
                            )}

                            {/* Sample Thumbnails */}
                            {thumbnails.length > 0 && (
                                <div>
                                    <h3 className="text-[10px] text-white/40 tracking-wider mb-3">SAMPLE_IMAGES</h3>
                                    <div className="grid grid-cols-3 gap-2">
                                        {thumbnails.map((thumb, i) => (
                                            <motion.div
                                                key={thumb.id}
                                                initial={{ opacity: 0, scale: 0.8 }}
                                                animate={{ opacity: 1, scale: 1 }}
                                                transition={{ delay: i * 0.1 }}
                                                whileHover={{ scale: 1.05, borderColor: 'rgba(255,255,255,0.5)' }}
                                                className="aspect-square border border-white/20 overflow-hidden cursor-pointer relative group"
                                            >
                                                <img
                                                    src={thumb.thumbnail}
                                                    alt={thumb.filename}
                                                    className="w-full h-full object-cover"
                                                />
                                                {thumb.class_name && (
                                                    <div className="absolute bottom-0 left-0 right-0 bg-black/70 px-1 py-0.5 text-[8px] text-white/70 opacity-0 group-hover:opacity-100 transition-opacity">
                                                        {thumb.class_name}
                                                    </div>
                                                )}
                                            </motion.div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Class Distribution */}
                            {distribution.length > 0 && (
                                <div>
                                    <h3 className="text-[10px] text-white/40 tracking-wider mb-3">CLASS_DISTRIBUTION</h3>
                                    <div className="space-y-2 max-h-[200px] overflow-y-auto">
                                        {distribution.map((item, i) => (
                                            <motion.div
                                                key={item.class_name}
                                                initial={{ opacity: 0, x: -20 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: i * 0.05 }}
                                                className="flex items-center gap-3"
                                            >
                                                <span className="text-[10px] text-white/60 w-20 truncate" title={item.class_name}>
                                                    {item.class_name}
                                                </span>
                                                <div className="flex-1 h-4 bg-white/5 relative overflow-hidden">
                                                    <motion.div
                                                        className="absolute left-0 top-0 h-full bg-white/30"
                                                        initial={{ width: 0 }}
                                                        animate={{ width: `${(item.count / maxCount) * 100}%` }}
                                                        transition={{ delay: i * 0.05 + 0.2, duration: 0.5 }}
                                                    />
                                                </div>
                                                <span className="text-[10px] text-white/40 w-16 text-right">
                                                    {item.count} ({item.percentage}%)
                                                </span>
                                            </motion.div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Warnings */}
                            {datasetDetails?.warnings?.length > 0 && (
                                <div className="p-3 border border-yellow-400/30 bg-yellow-400/10">
                                    <h3 className="text-[10px] text-yellow-400/70 tracking-wider mb-2">WARNINGS</h3>
                                    {datasetDetails.warnings.map((w, i) => (
                                        <p key={i} className="text-[11px] text-yellow-400/60">{w}</p>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </motion.div>
            </div>
        </div>
    )
}

export default Datasets
