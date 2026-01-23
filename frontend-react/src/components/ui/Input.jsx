import React from 'react';

const Input = ({ label, type = 'text', placeholder, value, onChange }) => (
  <div className="space-y-1.5">
    {label && <label className="text-sm font-medium text-slate-300">{label}</label>}
    <input
      type={type}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all"
    />
  </div>
);

export default Input;

