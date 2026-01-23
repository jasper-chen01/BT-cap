import React from 'react';

const Button = ({
  children,
  onClick,
  variant = 'primary',
  disabled,
  className = '',
  icon: Icon,
  type = 'button',
}) => {
  const baseStyle =
    'flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all duration-200 transform active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed';
  const variants = {
    primary:
      'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-900/20',
    secondary:
      'bg-slate-700 hover:bg-slate-600 text-slate-200 border border-slate-600',
    glass:
      'bg-white/10 hover:bg-white/20 text-white backdrop-blur-sm border border-white/10',
    outline:
      'border-2 border-indigo-500/50 text-indigo-400 hover:bg-indigo-500/10',
    ghost: 'bg-transparent hover:bg-slate-800 text-slate-300 hover:text-white',
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${baseStyle} ${variants[variant]} ${className}`}
    >
      {Icon && <Icon size={18} />}
      {children}
    </button>
  );
};

export default Button;

